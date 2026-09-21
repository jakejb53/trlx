"""Independent built-in evaluations; observations never control optimization."""

import contextlib
import copy
import hashlib
import json
import math
import os
import pathlib
import pickle
import random
import sys

from dataset.endpoint import Endpoint
from dataset.io import DatasetError, write_text
from dataset.progress import stage
from trlx import TrlxError, data_load, quality_scorers, show


FILENAME = show.QUALITY_FILENAME
SCORER_VERSION = 1


# Partial observations remain evidence even when a later scorer fails; they are never aggregated as a full round.
class EvaluationError(TrlxError):
    # Keep the ordinary human-readable error interface while retaining already observed samples.
    def __init__(self, message, results):
        super().__init__(message)
        self.results = results


# Validate the entire independent set before weights load; no benchmark rows are sampled away.
def load_data(settings, *, progress=None):
    dataset = data_load.load_ref(settings.quality_dataset, progress=progress)
    if not dataset.num_rows:
        raise TrlxError("[assessment].quality_dataset: the independent evaluation dataset is empty")
    with stage(progress, "validating independent quality dataset", total=dataset.num_rows, unit="rows") as activity:
        for index, row in enumerate(dataset, 1):
            quality_scorers.validate_row(settings.quality_preset, row, index)
            activity.advance()
    return dataset


# JSON metadata retains AddedToken matching flags rather than reducing tokens to strings.
def _identity_value(value):
    from tokenizers import AddedToken

    if isinstance(value, AddedToken):
        return value.__getstate__()
    if isinstance(value, dict):
        return {str(key): _identity_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"unsupported tokenizer metadata type {type(value).__module__}.{type(value).__qualname__}")


# Hash canonical codec metadata where available; arbitrary Python codecs require conservative full state.
def _processor_identity(processor):
    import tokenizers
    import transformers
    import trl

    tokenizer = getattr(processor, "tokenizer", processor)
    identity = {"class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
                "versions": {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                             "trl": trl.__version__}}
    backend = getattr(tokenizer, "backend_tokenizer", None)
    sentencepiece = getattr(tokenizer, "sp_model", None)
    if backend is not None:
        codec = json.loads(backend.to_str())
        # Quality encodes without padding/truncation; the backend stores the previous call's choices.
        codec["padding"] = codec["truncation"] = None
        identity["codec"] = codec
    elif sentencepiece is not None:
        import sentencepiece as sentencepiece_module

        identity["versions"]["sentencepiece"] = sentencepiece_module.__version__
        identity["codec"] = hashlib.sha256(sentencepiece.serialized_model_proto()).hexdigest()
    else:
        # Python codecs have no universal semantic serializer. Preserve all state rather than
        # silently weakening identity to vocabulary; cache changes may conservatively start a new series.
        identity["python_processor"] = hashlib.sha256(pickle.dumps(processor, protocol=5)).hexdigest()
        return identity

    # Mirror saved wrapper configuration, but read current attributes, not stale constructor values.
    ignored = {"name_or_path", "files_loaded", "tokenizer_file", "special_tokens_map_file", "cache_dir",
               "local_files_only", "is_local", "device_map", "token", "use_auth_token", "tokenizer_object",
               "__slow_tokenizer", "slow_tokenizer_class", *getattr(tokenizer, "vocab_files_names", {})}
    keys = (set(getattr(tokenizer, "init_kwargs", {})) - ignored) | {
        "model_max_length", "padding_side", "truncation_side", "model_input_names", "split_special_tokens",
        "clean_up_tokenization_spaces", "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output",
        "add_bos_token", "add_eos_token", "add_prefix_space", "legacy", "sp_model_kwargs",
        "special_tokens_pattern", "token_type_ids_pattern", "token_type_ids_include_special_tokens",
        "response_template", "chat_template", "special_tokens_map", "extra_special_tokens",
    }
    initial = getattr(tokenizer, "init_kwargs", {})
    wrapper = {key: getattr(tokenizer, key, initial.get(key)) for key in sorted(keys)}
    wrapper["added_tokens"] = {index: token for index, token in tokenizer.added_tokens_decoder.items()}
    identity["wrapper"] = _identity_value(wrapper)
    if processor is not tokenizer:
        # ProcessorMixin excludes its tokenizer and template from to_dict; both are retained separately.
        if hasattr(processor, "to_dict"):
            identity["processor"] = _identity_value(processor.to_dict())
            identity["processor_template"] = _identity_value(getattr(processor, "chat_template", None))
        else:
            identity["python_processor"] = hashlib.sha256(pickle.dumps(processor, protocol=5)).hexdigest()
    return identity


# Hash actual benchmark encodings too, including processor-specific transformations before model input.
def _encoded_identity(settings, processor, dataset):
    digest = hashlib.sha256()
    for row in dataset:
        if settings.quality_preset == "language_modeling":
            ids = [_lm_ids(processor, row)]
        elif settings.quality_preset == "preference":
            ids = [_preference_ids(processor, row, field) for field in ("chosen", "rejected")]
        else:
            ids = [_prompt_ids(processor, row, settings.quality_preset)]
        digest.update(json.dumps(ids, separators=(",", ":")).encode() + b"\n")
    return digest.hexdigest()


# Series identity includes full codec semantics and actual inputs, never a vocabulary-only approximation.
def series_id(settings, dataset_fingerprint, processor, model, *, dataset=None):
    try:
        processor = copy.deepcopy(processor)
        codec = _processor_identity(processor)
        encoded = _encoded_identity(settings, processor, dataset) if dataset is not None else None
    except Exception as error:
        raise TrlxError(f"cannot establish independent quality tokenizer identity: {type(error).__name__}: {error}") from error
    generation = getattr(model, "generation_config", None)
    judge = settings.judge
    identity = {
        "version": SCORER_VERSION, "preset": settings.quality_preset, "data": dataset_fingerprint,
        "max_length": settings.quality_max_length, "max_new_tokens": settings.quality_max_new_tokens,
        "batch_size": settings.quality_batch_size,
        "tokenizer": codec, "encoded_inputs": encoded, "chat_template": _template(processor),
        "generation": generation.to_dict() if generation is not None else None,
        "judge": {key: judge[key] for key in ("url", "model", "max_tokens")} if judge is not None else None,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# Inference and scoring may consume randomness, but training must observe its original streams.
@contextlib.contextmanager
def observational(model):
    import numpy as np
    import torch

    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.modules()]
    # PEFT delegates attributes and rebinds nested generation configs. Preserve every
    # actual owner, absence of wrapper-local attributes, and shared-object aliases.
    generation_owners = [(module, "generation_config" in vars(module), vars(module).get("generation_config"))
                         for module, _ in modes]
    try:
        copies = {}
        for module, owned, original in generation_owners:
            if owned:
                module.generation_config = copy.deepcopy(original, copies)
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        for module, owned, original in generation_owners:
            if owned:
                module.generation_config = original
            elif "generation_config" in vars(module):
                delattr(module, "generation_config")
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)


# A configured judge uses the existing endpoint client and shipped rubric, never a custom callback.
def judge_endpoint(settings):
    if settings.quality_preset not in {"instruction_following", "writing"}:
        return None
    spec = settings.judge
    name = spec["api_key"]
    key = os.environ.get(name) if name is not None else None
    if name is not None and not key:
        raise TrlxError(f"[assessment.judge].api_key: environment variable {name} is not set")
    return Endpoint(spec["url"], spec["model"], key, spec["timeout"], spec["retries"])


# A processor may own a template independently of its text tokenizer.
def _template(processor):
    tokenizer = getattr(processor, "tokenizer", processor)
    return getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)


# Match TRL's processor-aware chat normalization and batch unwrapping; raw LM text stays raw.
def _encode(processor, value, *, prompt=False):
    from trl.data_utils import _tokenize

    template = _template(processor)
    if prompt and isinstance(value, str) and template is not None:
        value = [{"role": "user", "content": value}]
    if isinstance(value, list):
        return _tokenize(processor, value, add_generation_prompt=prompt, chat_template=template)["input_ids"]
    return _tokenize(processor, value)["input_ids"]


# Both pre-run inspection and inference consume the identical full-sequence encoding.
def _lm_ids(processor, row):
    value = row.get("text", row.get("messages"))
    if value is None:
        value = row["prompt"] + row["completion"]
    return _encode(processor, value)


# RewardTrainer appends EOS to plain completions before whole-sequence tokenization.
def _preference_ids(processor, row, field):
    tokenizer = getattr(processor, "tokenizer", processor)
    value = row[field]
    if "prompt" in row:
        value = row["prompt"] + value
    if isinstance(value, str):
        if tokenizer.eos_token is None:
            raise TrlxError("plain preference quality rows require a tokenizer EOS token")
        if not value.endswith(tokenizer.eos_token):
            value += tokenizer.eos_token
    return _encode(processor, value)


# Preset constraints include choices/labels, never reference answers; limits measure the resulting prompt.
def _prompt_ids(processor, row, preset):
    return _encode(processor, quality_scorers.generation_prompt(preset, row), prompt=True)


# Scan the actual benchmark encodings before confirmation, using the same encoders as inference.
def inspect_inputs(settings, processor, dataset, model_metadata, *, progress=None):
    processor = copy.deepcopy(processor)
    preset = settings.quality_preset
    lengths, invalid, over_limit, budgets = [], [], [], []
    with stage(progress, "scanning independent quality inputs", total=dataset.num_rows, unit="rows") as activity:
        for number, row in enumerate(dataset, 1):
            try:
                if preset == "language_modeling":
                    counts = [len(_lm_ids(processor, row))]
                    if counts[0] < 2:
                        raise TrlxError("fewer than two tokens are available for language-modeling loss")
                    budget = min(counts[0], settings.quality_max_length)
                elif preset == "preference":
                    counts = [len(_preference_ids(processor, row, field)) for field in ("chosen", "rejected")]
                    budget = max(counts)
                else:
                    counts = [len(_prompt_ids(processor, row, preset))]
                    budget = counts[0] + settings.quality_max_new_tokens
                lengths.append({"row": number, "tokens": counts})
                budgets.append(budget)
                if preset != "language_modeling" and (min(counts) == 0 or max(counts) > settings.quality_max_length):
                    over_limit.append(number)
            except Exception as error:
                # Failed inspection is unavailable evidence, never a fabricated zero-token sample.
                invalid.append({"row": number, "type": type(error).__name__, "error": str(error)})
            activity.advance()
    findings = []
    if invalid or over_limit:
        findings.append({"code": "quality_input_limits", "severity": "warning", "basis": "measured",
                         "summary": "Some independent quality inputs cannot be scored with the selected tokenizer and input limit.",
                         "evidence": {"inspection_errors": invalid, "over_limit_rows": over_limit,
                                      "quality_max_length": settings.quality_max_length},
                         "recommendation": "Correct the identified rows or increase the explicit quality input limit within model capacity; examples will not be silently discarded."})
    maximum = model_metadata.get("max_position_embeddings")
    if type(maximum) in (int, float) and math.isfinite(maximum) and maximum > 0 and budgets and max(budgets) > maximum:
        findings.append({"code": "quality_context_budget", "severity": "warning", "basis": "projected",
                         "summary": "The independent quality sequence budget exceeds the model's declared position count.",
                         "evidence": {"largest_sequence_budget": max(budgets), "max_position_embeddings": maximum,
                                      "rope_scaling": model_metadata.get("rope_scaling")},
                         "recommendation": "Check documented context/scaling support and adjust quality input or generation limits before enabling expensive checks."})
    return {"input_lengths": lengths, "input_errors": invalid, "over_limit_rows": over_limit,
            "largest_sequence_budget": max(budgets) if budgets else None, "findings": findings}


# Agree on actual inputs before model collectives: stochastic/custom tokenizers must not split rank control flow.
def _coordinated_ids(processor, row, settings, field=None):
    import torch.distributed as dist

    ids, error, digest = None, None, None
    try:
        if settings.quality_preset == "language_modeling":
            ids = _lm_ids(processor, row)
            if len(ids) < 2:
                raise TrlxError("language-modeling quality row has fewer than two tokens to score")
        elif settings.quality_preset == "preference":
            ids = _preference_ids(processor, row, field)
        else:
            ids = _prompt_ids(processor, row, settings.quality_preset)
        if settings.quality_preset != "language_modeling" and (not ids or len(ids) > settings.quality_max_length):
            raise TrlxError(f"quality {field or 'prompt'} is empty or exceeds quality_max_length; no input was silently truncated")
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    statuses = [(error, digest)]
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        statuses = [None] * dist.get_world_size()
        dist.all_gather_object(statuses, (error, digest))
    errors = [f"rank {index}: {message}" for index, (message, _) in enumerate(statuses) if message is not None]
    if errors:
        raise TrlxError("independent quality input unavailable: " + "; ".join(errors))
    if len({value for _, value in statuses}) != 1:
        raise TrlxError("independent quality tokenization differs between ranks; model forwards were not started")
    return ids


# Full-sequence LM evaluation uses overlapping windows so only the corpus's first token is unscored.
def _language_modeling(model, processor, row, settings):
    import torch
    import torch.nn.functional as functional

    ids = _coordinated_ids(processor, row, settings)
    device = next(model.parameters()).device
    total_loss, tokens, windows = 0.0, 0, 0
    stride = settings.quality_max_length - 1
    for start in range(0, len(ids) - 1, stride):
        inputs = torch.tensor([ids[start:start + settings.quality_max_length]], device=device)
        outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs))
        if outputs.logits is None:
            raise TrlxError("model returned no logits for independent language-modeling evaluation")
        # Tiling the reduction avoids a second full-vocabulary float32 allocation.
        for offset in range(0, inputs.shape[1] - 1, 64):
            stop = min(offset + 64, inputs.shape[1] - 1)
            total_loss += float(functional.cross_entropy(outputs.logits[0, offset:stop].float(),
                                                        inputs[0, offset + 1:stop + 1], reduction="sum"))
        tokens += inputs.shape[1] - 1
        windows += 1
    loss = total_loss / tokens
    perplexity = math.exp(loss) if loss < math.log(sys.float_info.max) else None
    return {"score": None, "metrics": {"loss": loss, **({"perplexity": perplexity} if perplexity is not None else {})},
            "details": {"nll_sum": total_loss, "tokens": tokens, "windows": windows,
                        "scope": "full sequence; one-token overlap between context windows",
                        "perplexity_overflow": perplexity is None}}


# Reward-model quality asks whether the supplied preference is ranked correctly, including ties.
def _preference(model, processor, row, settings):
    import torch

    values = []
    for field in ("chosen", "rejected"):
        ids = _coordinated_ids(processor, row, settings, field)
        inputs = torch.tensor([ids], device=next(model.parameters()).device)
        logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits
        if logits.numel() != 1:
            raise TrlxError("preference quality evaluation requires a scalar reward-model output")
        values.append(float(logits.item()))
    if not all(math.isfinite(value) for value in values):
        raise TrlxError("preference quality evaluation returned a non-finite reward score")
    margin = values[0] - values[1]
    return {"score": float(margin > 0),
            "metrics": {"accuracy": float(margin > 0), "tie_rate": float(margin == 0), "margin": margin},
            "details": {"chosen_score": values[0], "rejected_score": values[1]}}


# PEFT delegates generation below the FSDP root. Register this entry point only
# for the observation, and defer generation exceptions until the post-hook runs.
def _generate(model, **kwargs):
    from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method

    if not isinstance(model, FSDPModule):
        return model.generate(**kwargs)
    owned = "generate" in vars(model)
    original = model.generate
    failure = []

    # PyTorch's method wrapper has no finally: returning normally here lets its
    # post-forward hook release parameters and reset state before we re-raise.
    def guarded(**arguments):
        try:
            return original(**arguments)
        except BaseException as error:
            failure.append((error, error.__traceback__))
            return None

    try:
        model.generate = guarded
        register_fsdp_forward_method(model, "generate")
        result = model.generate(**kwargs)
        if failure:
            error, traceback = failure[0]
            raise error.with_traceback(traceback)
        return result
    finally:
        if owned:
            model.generate = original
        else:
            del model.generate


# Decode the generated suffix only and retain whether it exhausted the configured budget without EOS.
def _generate_batch(model, processor, rows, settings, distributed):
    import torch

    tokenizer = getattr(processor, "tokenizer", processor)
    sequences = []
    for row in rows:
        sequences.append(_coordinated_ids(processor, row, settings))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise TrlxError("independent generation needs a tokenizer pad or EOS token")
    width = max(map(len, sequences))
    device = next(model.parameters()).device
    inputs = torch.tensor([[pad] * (width - len(ids)) + ids for ids in sequences], device=device)
    masks = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in sequences], device=device)
    outputs = _generate(model, input_ids=inputs, attention_mask=masks, max_new_tokens=settings.quality_max_new_tokens,
                             do_sample=False, num_beams=1, num_return_sequences=1, use_cache=False,
                             pad_token_id=pad, synced_gpus=distributed, return_dict_in_generate=False,
                             output_scores=False, output_logits=False, output_attentions=False, output_hidden_states=False)
    eos = getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id)
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    result = []
    for generated in outputs[:, width:]:
        ids = generated.tolist()
        terminated = any(token in eos for token in ids)
        result.append((tokenizer.decode(ids, skip_special_tokens=True),
                       len(ids) >= settings.quality_max_new_tokens and not terminated))
    return result


# A rank-zero scorer failure must be communicated before any rank enters the next model collective.
def _broadcast(value, distributed):
    if distributed:
        import torch.distributed as dist

        values = [value]
        dist.broadcast_object_list(values, src=0)
        return values[0]
    return value


# Every rank performs identical forwards; only rank zero calls judges and retains/publishes evidence.
def evaluate(model, processor, dataset, settings, *, rank=0, progress=None):
    import torch.distributed as dist

    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    processor = copy.deepcopy(processor)
    preset = settings.quality_preset
    endpoint, setup_error = None, None
    if rank == 0:
        try:
            endpoint = judge_endpoint(settings)
        except (TrlxError, DatasetError) as error:
            setup_error = str(error)
    setup_error = _broadcast(setup_error, distributed)
    if setup_error is not None:
        raise TrlxError(setup_error)
    results = []
    with observational(model), stage(progress, "independent quality evaluation", total=dataset.num_rows, unit="rows") as activity:
        for start in range(0, dataset.num_rows, settings.quality_batch_size):
            rows = [dataset[index] for index in range(start, min(start + settings.quality_batch_size, dataset.num_rows))]
            if preset in {"language_modeling", "preference"}:
                function = _language_modeling if preset == "language_modeling" else _preference
                scores = [function(model, processor, row, settings) for row in rows]
                outputs = [(None, False)] * len(rows)
            else:
                outputs = _generate_batch(model, processor, rows, settings, distributed)
                scores = [None] * len(rows)
            error = None
            if rank == 0:
                try:
                    for index, (row, (output, truncated), scored) in enumerate(zip(rows, outputs, scores), start + 1):
                        reply = None
                        if scored is None:
                            if endpoint is not None:
                                reply = endpoint.complete(quality_scorers.judge_messages(preset, row, output),
                                                          max_tokens=settings.judge["max_tokens"], progress=activity)
                                scored = quality_scorers.parse_judge_reply(reply)
                                scored["details"]["judge_reply"] = reply
                            else:
                                scored = quality_scorers.score_generation(preset, row, output)
                            scored["metrics"].update(quality_scorers.generation_diagnostics(output, truncated=truncated))
                        results.append({"row": index, "input": row, "output": output, **scored})
                except (TrlxError, DatasetError, ValueError) as exc:
                    error = str(exc)
                    failed = {"row": index, "input": row, "output": output, "error": error}
                    if reply is not None:
                        failed["judge_reply"] = reply
                    results.append(failed)
            error = _broadcast(error, distributed)
            if error is not None:
                raise EvaluationError(f"quality rows {start + 1}-{start + len(rows)}: {error}", results)
            activity.advance(len(rows))
    return results


# Corpus NLL is token-weighted; other built-in per-example metrics retain explicit denominators.
def aggregate(preset, results):
    if not results:
        raise TrlxError("independent quality evaluation produced no scored rows")
    metrics = {"quality/rows": len(results)}
    if preset == "language_modeling":
        tokens = sum(row["details"]["tokens"] for row in results)
        loss = sum(row["details"]["nll_sum"] for row in results) / tokens
        metrics.update({"quality/loss": loss, "quality/tokens": tokens})
        if loss < math.log(sys.float_info.max):
            metrics["quality/perplexity"] = math.exp(loss)
        return metrics
    names = {name for row in results for name in row["metrics"]}
    for name in sorted(names):
        values = [row["metrics"][name] for row in results if name in row["metrics"]]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise TrlxError(f"independent quality metric {name} contains a non-finite or nonnumeric value")
        metrics["quality/" + name] = sum(values) / len(values)
        metrics["quality/metric_rows/" + name] = len(values)
    scores = [row["score"] for row in results if row["score"] is not None]
    if scores:
        metrics["quality/score"] = sum(scores) / len(scores)
        metrics["quality/scored_rows"] = len(scores)
    if preset in {"classification", "multiple_choice"}:
        from urllib.parse import quote

        # Support counts accompany each class result; class imbalance cannot hide behind accuracy.
        labels = {row["details"]["expected_key"] for row in results}
        for label in sorted(labels):
            members = [row for row in results if row["details"]["expected_key"] == label]
            prefix = "quality/class/" + quote(label, safe="")
            metrics[prefix + "/rows"] = len(members)
            metrics[prefix + "/accuracy"] = sum(row["score"] for row in members) / len(members)
    return metrics


# Publish complete rounds: raw samples live here, while aggregate metrics have one metrics.jsonl writer.
def publish(run_dir, context, results, *, no_staging=False, progress=None):
    path = pathlib.Path(run_dir) / FILENAME
    try:
        previous = path.read_text(encoding="utf-8") if path.exists() else ""
        content = json.dumps({"quality": context, "results": results}, ensure_ascii=False, allow_nan=False) + "\n"
        write_text(path, previous + content, force=True, no_staging=no_staging, progress=progress)
    except (OSError, UnicodeError, ValueError, DatasetError) as error:
        raise TrlxError(f"{path}: cannot publish independent quality evidence: {error}") from error


# Build lazily so reading stored results never imports the training stack.
def callback_class():
    from transformers import TrainerCallback

    class QualityCallback(TrainerCallback):
        # Rank zero owns publication; every rank must take the same evaluation branches.
        def __init__(self, settings, run_dir, writer, rank, *, no_staging=False, progress=None):
            self.settings = settings
            self.run_dir = pathlib.Path(run_dir)
            self.writer = writer
            self.rank = rank
            self.no_staging = no_staging
            self.progress = progress
            self.trainer = None
            self.dataset = None
            self.series = None

        # The trainer owns wrapping and processor adjustments; observe their final objects.
        def bind(self, trainer):
            self.trainer = trainer

        # Each rank validates its input before any model collective, avoiding divergent row loops.
        def _prepare(self, model):
            import torch.distributed as dist
            from trlx import data_profile

            local_error = None
            try:
                if self.dataset is None:
                    dataset = load_data(self.settings, progress=self.progress)
                    series = series_id(self.settings, data_profile.fingerprint(dataset),
                                       self.trainer.processing_class, model, dataset=dataset)
                    # Cache only a completely validated dataset/series pair; failed setup is retried at completion.
                    self.dataset, self.series = dataset, series
            except (TrlxError, DatasetError, OSError, ValueError) as error:
                local_error = str(error)
            states = [(local_error, self.series)]
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                states = [None] * dist.get_world_size()
                dist.all_gather_object(states, (local_error, self.series))
            errors = [f"rank {index}: {error}" for index, (error, _) in enumerate(states) if error is not None]
            if errors:
                raise TrlxError("; ".join(errors))
            if len({series for _, series in states}) != 1:
                raise TrlxError("independent quality data or tokenizer differs between training ranks")

        # The starting comparison is at step zero, or the resumed step for a new evaluation series.
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self._run("baseline", args, state, model)

        # Follow ordinary evaluation events, including operator-selected step or epoch schedules.
        def on_evaluate(self, args, state, control, model=None, **kwargs):
            self._run("scheduled", args, state, model)

        # Completion is unconditional when checks are enabled, even with eval_strategy='no'.
        def on_train_end(self, args, state, control, model=None, **kwargs):
            self._run("completion", args, state, model)

        # Preserve all training state around preparation, inference, scoring, and reporting.
        def _run(self, phase, args, state, model):
            import torch.distributed as dist

            if self.trainer is None:
                raise RuntimeError("quality callback was not bound to its trainer")
            base = self.trainer.accelerator.unwrap_model(model)
            distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
            context = {"phase": phase, "step": state.global_step, "preset": self.settings.quality_preset,
                       "series": self.series, "status": "failed"}
            # A final load-best-model action belongs to the trainer, and must be visible in the evidence.
            if phase == "completion" and args.load_best_model_at_end:
                context["model_checkpoint"] = state.best_model_checkpoint
            with observational(base):
                try:
                    self._prepare(base)
                    context.update(series=self.series, rows=self.dataset.num_rows)
                    retained = self.writer.has_baseline(self.series) if self.rank == 0 else None
                    if phase == "baseline" and _broadcast(retained, distributed):
                        if self.rank == 0:
                            print("independent quality: retained matching baseline from this run", flush=True)
                        return
                    results = evaluate(base, self.trainer.processing_class, self.dataset, self.settings,
                                       rank=self.rank, progress=self.progress)
                    publication_error = None
                    if self.rank == 0:
                        try:
                            values = aggregate(self.settings.quality_preset, results)
                            context["status"] = "complete"
                            publish(self.run_dir, context, results, no_staging=self.no_staging, progress=self.progress)
                            self.writer.quality(args, state, values, context)
                            primary = values.get("quality/score", values.get("quality/loss"))
                            print(f"independent quality {phase}: {self.dataset.num_rows} rows; "
                                  f"{'score' if 'quality/score' in values else 'loss'}={primary}; "
                                  f"evidence: {self.run_dir / FILENAME}", flush=True)
                        except (TrlxError, DatasetError, OSError, ValueError) as error:
                            publication_error = str(error)
                    publication_error = _broadcast(publication_error, distributed)
                    if publication_error is not None:
                        raise TrlxError(publication_error)
                except (TrlxError, DatasetError, OSError, ValueError) as error:
                    # Expected evaluation/scoring failures are unavailable evidence, never a zero score or stop signal.
                    if self.rank == 0:
                        context.update(status="failed", error=str(error))
                        print(f"independent quality {phase} failed: {error}; training settings and control are unchanged", flush=True)
                        try:
                            publish(self.run_dir, context, getattr(error, "results", []),
                                    no_staging=self.no_staging, progress=self.progress)
                        except TrlxError as publication_error:
                            print(f"independent quality evidence unavailable: {publication_error}", flush=True)

    return QualityCallback
