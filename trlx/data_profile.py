"""Full, weightless data assessment; preparation remains the trainer's responsibility.

Source identities are measured. Token counts and filtering/packing consequences
are projections of the installed TRL contract, never replacement training data.
The caller supplies the configured processor (including chat-template overrides).
"""

import collections
import copy
import hashlib
import json
import math

from datasets import Dataset
from trl import pack_dataset
from trl.chat_template_utils import get_training_chat_template, has_generation_markers
from trl.data_utils import _tokenize, extract_prompt, is_conversational, maybe_convert_to_chatml

from dataset.progress import stage
from trlx import chat_encoding


# Stable identity compares source content, not dictionary insertion order.
def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


# Hashes keep large examples out of the evidence artifact without losing indices.
def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


# The same ordered source fingerprint is shared with quality series identity.
# Unsupported payloads fail explicitly instead of hashing process-specific repr.
def fingerprint(dataset, *, exclude_columns=()):
    digest = hashlib.sha256()
    for index, original in enumerate(dataset):
        try:
            row = {key: value for key, value in original.items() if key not in exclude_columns}
            digest.update(bytes.fromhex(_digest(row)))
        except (ValueError, TypeError) as error:
            raise ValueError(f"cannot fingerprint dataset row {index}: {error}") from error
    return digest.hexdigest()


# Percentiles use linear interpolation over the complete observed population.
def _distribution(values):
    ordered = sorted(values)
    if not ordered:
        return dict.fromkeys(("min", "max", "mean", "p50", "p95", "p99"))
    result = {"min": ordered[0], "max": ordered[-1], "mean": sum(ordered) / len(ordered)}
    for name, fraction in (("p50", .5), ("p95", .95), ("p99", .99)):
        position = fraction * (len(ordered) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        result[name] = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return result


# Template tools may be JSON-encoded columns; use the same interpretation as TRL.
def _template_kwargs(row):
    tools = row.get("tools")
    return {"tools": json.loads(tools) if isinstance(tools, str) else tools,
            **(row.get("chat_template_kwargs") or {})}


# EOS is appended to plain-text targets only, as in TRL dataset preparation.
def _eos(text, tokenizer):
    if tokenizer.eos_token is None:
        raise ValueError("plain-text preparation requires a tokenizer EOS token")
    return text if text.endswith(tokenizer.eos_token) else text + tokenizer.eos_token


# The tokenizer's raw prefix can change at a text boundary. TRL still slices by
# its length; report that mismatch instead of silently repairing trainer inputs.
def _prompt_completion(processor, row, key, *, template=None, assistant=False):
    conversational = is_conversational(row)
    kwargs = {"chat_template": template, **_template_kwargs(row)} if conversational else {}
    prompt = _tokenize(processor, row["prompt"],
                       **({"add_generation_prompt": True, **kwargs} if conversational else {}))["input_ids"]
    target = row[key] if conversational else _eos(row[key], getattr(processor, "tokenizer", processor))
    complete = _tokenize(processor, row["prompt"] + target,
                         **({"return_assistant_tokens_mask": assistant, **kwargs} if conversational else {}))
    return prompt, complete, complete["input_ids"][:len(prompt)] != prompt


# Labels supplied by the operator are authoritative; optional masks are folded
# only when labels are absent, matching SFT's preparation and skip behavior.
def _sft_row(row, processor, args, template, completion_only):
    skip = (getattr(args, "dataset_kwargs", None) or {}).get("skip_prepare_dataset", False)
    if skip and "input_ids" not in row:
        raise ValueError("skip_prepare_dataset requires trainer-ready input_ids")
    mismatch = False
    if "input_ids" in row:
        encoded = row
        if skip and "labels" not in row and ("completion_mask" in row or "assistant_masks" in row):
            raise ValueError("skip_prepare_dataset skips loss-mask conversion; supply labels")
    else:
        row = maybe_convert_to_chatml(copy.deepcopy(row))
        if "prompt" in row:
            prompt, encoded, mismatch = _prompt_completion(
                processor, row, "completion", template=template, assistant=args.assistant_only_loss)
            encoded["completion_mask"] = [0] * len(prompt) + [1] * (len(encoded["input_ids"]) - len(prompt))
        elif is_conversational(row):
            encoded = _tokenize(processor, row["messages"], return_assistant_tokens_mask=args.assistant_only_loss,
                                **{"chat_template": template, **_template_kwargs(row)})
        else:
            # TRL only appends EOS to its conventional text column, even when
            # dataset_text_field selects a different operator-provided column.
            value = row[args.dataset_text_field]
            if args.dataset_text_field == "text":
                value = _eos(value, getattr(processor, "tokenizer", processor))
            encoded = _tokenize(processor, value)
        if "assistant_masks" in encoded and 1 not in encoded["assistant_masks"]:
            raise ValueError("assistant_only_loss produced no assistant tokens before truncation")
        # Dataset.map preserves original columns not replaced by tokenization.
        encoded = {**row, **encoded}
    ids = list(encoded["input_ids"])
    if "labels" in encoded:
        labels = list(encoded["labels"])
    else:
        masks = []
        if not skip:
            if completion_only and "completion_mask" in encoded:
                masks.append(encoded["completion_mask"])
            if "assistant_masks" in encoded:
                masks.append(encoded["assistant_masks"])
        labels = [token if all(mask[index] for mask in masks) else -100 for index, token in enumerate(ids)]
    if len(labels) != len(ids):
        raise ValueError("labels and input_ids must have the same length")
    record = {"input_ids": ids, "labels": labels}
    if skip and "seq_lengths" in row:
        record["seq_lengths"] = list(row["seq_lengths"])
    return record, mismatch


# Return complete branch sequences plus masks. DPO/KTO concatenate the separately
# tokenized prompt with the sliced response, even when token boundaries differ.
def _preference_row(row, processor, method):
    if method == "reward":
        keys = ("chosen_ids", "rejected_ids")
        if all(key in row for key in keys):
            return [(list(row[key]), None) for key in keys], None, False
        legacy = ("chosen_input_ids", "rejected_input_ids")
        if all(key in row for key in legacy):
            return [(list(row[key]), None) for key in legacy], None, False
        branches = []
        conversational = is_conversational(row)
        kwargs = _template_kwargs(row) if conversational else {}
        for key in ("chosen", "rejected"):
            target = row[key] if conversational else _eos(row[key], getattr(processor, "tokenizer", processor))
            value = row["prompt"] + target if "prompt" in row else target
            branches.append((_tokenize(processor, value, **kwargs)["input_ids"], None))
        return branches, None, False
    if "prompt" not in row:
        row = {**row, **extract_prompt(row)}
    branches, mismatch, prompt_length = [], False, None
    for key in (("chosen", "rejected") if method == "dpo" else ("completion",)):
        prompt, encoded, changed = _prompt_completion(processor, row, key)
        response = encoded["input_ids"][len(prompt):]
        branches.append((prompt + response, [0] * len(prompt) + [1] * len(response)))
        mismatch |= changed
        prompt_length = len(prompt)
    return branches, prompt_length, mismatch


# Off-policy preflight must score the exact response IDs that DPO/KTO train on.
# In particular, independent response tokenization can disagree at BPE boundaries.
def response_pairs(processor, row, method):
    if method not in {"dpo", "kto"}:
        raise ValueError(f"response_pairs requires dpo or kto, got {method!r}")
    branches, prompt_length, _ = _preference_row(row, processor, method)
    return [(ids[:prompt_length], ids[prompt_length:]) for ids, _ in branches]


# The same causal shift applies to every sequence. Padding-free BFD additionally
# masks the first token of each packed document, not only the first packed token.
def _loss_positions(row, padding_free):
    labels = row["labels"]
    starts = {0}
    if padding_free and "seq_lengths" in row:
        offset = 0
        for length in row["seq_lengths"]:
            starts.add(offset)
            offset += length
    return [index for index, label in enumerate(labels) if label != -100 and index not in starts]


# Identity analysis is independent of tokenization success and covers every row.
def _identities(rows, method, *, exclude_columns=()):
    examples, prompts, pairs, labels = (collections.defaultdict(list) for _ in range(4))
    source_digest = hashlib.sha256()
    shapes = collections.Counter()
    identical_preferences = []
    identity_errors = []
    for index, original in enumerate(rows):
        row = {key: value for key, value in original.items() if key not in exclude_columns}
        try:
            digest = _digest(row)
        except (ValueError, TypeError) as error:
            identity_errors.append({"row": index, "message": f"cannot fingerprint row: {error}"})
            continue
        source_digest.update(bytes.fromhex(digest))
        examples[digest].append(index)
        shape = "pretokenized" if "input_ids" in row or "chosen_ids" in row else (
            "conversational" if is_conversational(row) else "plain_text")
        shapes[shape] += 1
        if "chosen" in row and "rejected" in row:
            try:
                normalized = row if "prompt" in row else {**row, **extract_prompt(row)}
            except (TypeError, ValueError, IndexError, KeyError) as error:
                identity_errors.append({"row": index, "message": f"cannot identify preference prompt: {error}"})
                continue
            prompt = normalized["prompt"]
            chosen, rejected = _digest(normalized["chosen"]), _digest(normalized["rejected"])
            key = (_digest(prompt), min(chosen, rejected), max(chosen, rejected))
            pairs[key].append((index, chosen <= rejected))
            if chosen == rejected:
                identical_preferences.append(index)
        else:
            prompt = row.get("prompt")
        if prompt is not None:
            prompts[_digest(prompt)].append(index)
        if "label" in row and "completion" in row:
            labels[_digest([row.get("prompt"), row["completion"]])].append((index, row["label"]))
    conflicting = [[index for index, _ in group] for group in (*pairs.values(), *labels.values())
                   if len({_canonical(label) for _, label in group}) > 1]
    return {"fingerprint": None if identity_errors else source_digest.hexdigest(), "shapes": dict(shapes),
            "identity_errors": identity_errors,
            "duplicates": [indices for indices in examples.values() if len(indices) > 1],
            "conflicting_labels": conflicting, "identical_preference_rows": identical_preferences}, examples, prompts


# Comparison preserves every matching source index. An empty exact-match result
# establishes neither semantic independence nor absence of paraphrased leakage.
def _overlap(train_examples, train_prompts, other_examples, other_prompts):
    return {name: [{"train_rows": left[key], "eval_rows": right[key]} for key in left if key in right]
            for name, left, right in (("identical_examples", train_examples, other_examples),
                                      ("prompts", train_prompts, other_prompts))}


# Quality data can have a different schema; source comparison needs no tokenizer
# or trainer preparation. Marker exclusions must identify known internal fields.
def compare_sources(train_set, other_set, method, *, train_exclude_columns=(), other_exclude_columns=()):
    train, train_examples, train_prompts = _identities(train_set, method, exclude_columns=train_exclude_columns)
    other, other_examples, other_prompts = _identities(other_set, method, exclude_columns=other_exclude_columns)
    return {**_overlap(train_examples, train_prompts, other_examples, other_prompts),
            "basis": "measured", "scope": "Exact whole examples and exact prompts only; no semantic independence claim.",
            "identity_errors": {"train": train["identity_errors"], "other": other["identity_errors"]}}


# Embedded image/video/audio blocks also require processor expansion, even when
# no top-level media column is present. Text-only blocks remain inspectable.
def _multimodal(row):
    if any(key in row for key in ("image", "images", "video", "videos", "audio")):
        return True
    for key in ("messages", "prompt", "completion", "chosen", "rejected"):
        value = row.get(key)
        if isinstance(value, list):
            for message in value:
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, list) and any(part.get("type") != "text" for part in content):
                    return True
    return False


# Full split projection never silently samples or suppresses invalid rows.
def _split(cfg, processor, rows, name, progress, template, completion_only):
    args, method = cfg.args, cfg.method.name
    # Only replay KL introduces this marker. With KL off, a source's own replay
    # column remains ordinary semantic content and participates in its identity.
    replay = getattr(cfg, "replay", None)
    excluded = ("replay",) if name == "train" and replay is not None and replay.kl_coef > 0 else ()
    identity, examples, prompts = _identities(rows, method, exclude_columns=excluded)
    result = {"rows": len(rows), **identity, "columns": sorted(set().union(*(row.keys() for row in rows))),
              "basis": "projected", "truncated_rows": [], "dropped_rows": [], "zero_loss_rows": [],
              "boundary_mismatch_rows": [], "errors": [], "unknowns": [], "raw_tokens": 0,
              "retained_tokens": 0, "discarded_tokens": 0, "loss_tokens": 0, "effective_rows": 0}
    raw_lengths, prepared_lengths, prompt_lengths, completion_lengths = [], [], [], []
    records, per_row = [], []
    class_counts = collections.Counter()
    skip = (getattr(args, "dataset_kwargs", None) or {}).get("skip_prepare_dataset", False)
    packing = getattr(args, "packing", False)
    if name == "eval" and getattr(args, "eval_packing", None) is not None:
        packing = args.eval_packing
    packing = method == "sft" and packing and not skip
    maximum = getattr(args, "max_length", None)
    mode = getattr(args, "truncation_mode", "keep_start") if method != "kto" else "keep_start"
    window = slice(-maximum, None) if maximum is not None and mode == "keep_end" else slice(None, maximum)
    with stage(progress, f"assessing all {name} rows", total=len(rows), unit="rows") as activity:
        for index, row in enumerate(rows):
            detail = {"raw_tokens": 0, "retained_tokens": 0, "loss_tokens": 0, "effective_rows": 0}
            try:
                if _multimodal(row):
                    raise ValueError("multimodal expanded token counts require the trainer processor/collator")
                if method == "sft":
                    record, mismatch = _sft_row(row, processor, args, template, completion_only)
                    length = len(record["input_ids"])
                    raw_lengths.append(length)
                    detail["raw_tokens"] = length
                    if not packing and not skip:
                        record = {key: value[window] for key, value in record.items()}
                        if length > len(record["input_ids"]):
                            result["truncated_rows"].append(index)
                        if maximum is not None and not any(label != -100 for label in record["labels"]):
                            result["dropped_rows"].append(index)
                            record = None
                    if record is not None:
                        if packing and args.packing_strategy == "bfd" and length > maximum:
                            result["truncated_rows"].append(index)
                        # A parallel token column follows the SAME packing,
                        # shuffling and overflow slices as labels. It belongs
                        # only to this projection and never enters training data.
                        record["_source_rows"] = [index] * len(record["input_ids"])
                        records.append(record)
                        detail["effective_rows"] = 1
                elif method in {"dpo", "kto", "reward"}:
                    branches, prompt_length, mismatch = _preference_row(row, processor, method)
                    lengths = [len(ids) for ids, _ in branches]
                    raw_lengths.extend(lengths)
                    detail["raw_tokens"] = sum(lengths)
                    if prompt_length is not None:
                        prompt_lengths.append(prompt_length)
                        completion_lengths.extend(length - prompt_length for length in lengths)
                    drop = maximum is not None and (
                        any(length > maximum for length in lengths) if method == "reward" else
                        mode == "keep_start" and prompt_length >= maximum)
                    if drop:
                        result["dropped_rows"].append(index)
                    else:
                        kept = branches if method == "reward" else [(ids[window], mask[window]) for ids, mask in branches]
                        retained = [len(ids) for ids, _ in kept]
                        prepared_lengths.extend(retained)
                        detail.update(retained_tokens=sum(retained), effective_rows=1)
                        detail["loss_tokens"] = (None if method == "reward" else
                                                 sum(sum(mask[1:]) for _, mask in kept))
                        if retained != lengths:
                            result["truncated_rows"].append(index)
                        if detail["loss_tokens"] == 0:
                            result["zero_loss_rows"].append(index)
                        if method == "kto":
                            class_counts["desirable" if row["label"] else "undesirable"] += 1
                else:
                    kwargs = getattr(args, "chat_template_kwargs", None) or {}
                    ids = _tokenize(processor, row["prompt"], add_generation_prompt=True, **kwargs)["input_ids"]
                    length = len(ids)
                    raw_lengths.append(length)
                    prompt_lengths.append(length)
                    prepared_lengths.append(length)
                    detail.update(raw_tokens=length, retained_tokens=length, effective_rows=1, loss_tokens=None)
                    mismatch = False
                if mismatch:
                    result["boundary_mismatch_rows"].append(index)
            except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
                result["errors"].append({"row": index, "message": str(error)})
            per_row.append(detail)
            activity.advance()
    if method == "sft":
        if packing and records:
            # Use public packing so binning, split boundaries and the library's
            # map batch/worker partitions match preparation, rather than a ratio.
            prepared = Dataset.from_list(records)
            if args.shuffle_dataset:
                prepared = prepared.shuffle(seed=args.seed, keep_in_memory=True)
            prepared = pack_dataset(prepared, maximum, args.packing_strategy,
                                    {"num_proc": args.dataset_num_proc, "keep_in_memory": True})
            records = list(prepared)
        # The collator is configured from TRAIN packing even when eval_packing
        # differs or preparation is skipped; both BFD variants reset documents.
        padding_free = args.padding_free or (args.packing and args.packing_strategy in {"bfd", "bfd_split"})
        # Count targets after packed document/fragment starts are masked. Wrapped
        # packing can activate an original row's first label; BFD split can mask
        # an interior label at the beginning of a newly created fragment.
        for record in records:
            for source in record["_source_rows"]:
                per_row[source]["retained_tokens"] += 1
            for position in _loss_positions(record, padding_free):
                per_row[record["_source_rows"][position]]["loss_tokens"] += 1
        result["zero_loss_rows"] = [index for index, detail in enumerate(per_row)
                                    if detail["effective_rows"] and not detail["loss_tokens"]]
        prepared_lengths = [len(record["input_ids"]) for record in records]
        result.update(effective_rows=len(records), retained_tokens=sum(prepared_lengths),
                      loss_tokens=sum(detail["loss_tokens"] for detail in per_row))
        result["packing"] = {"enabled": packing, "strategy": args.packing_strategy if packing else None,
                             "padding_free": padding_free}
    else:
        result["effective_rows"] = sum(row["effective_rows"] for row in per_row)
        result["retained_tokens"] = sum(row["retained_tokens"] for row in per_row)
        result["loss_tokens"] = (None if method in {"reward", "grpo", "rloo", "distillation"}
                                 else sum(row["loss_tokens"] for row in per_row))
    result["raw_tokens"] = sum(row["raw_tokens"] for row in per_row)
    result["discarded_tokens"] = result["raw_tokens"] - result["retained_tokens"]
    result.update(lengths=_distribution(raw_lengths), prepared_lengths=_distribution(prepared_lengths),
                  prompt_lengths=_distribution(prompt_lengths), completion_lengths=_distribution(completion_lengths))
    if method == "kto":
        result["label_counts"] = {"desirable": class_counts["desirable"], "undesirable": class_counts["undesirable"]}
    if method in {"grpo", "rloo", "distillation"}:
        result["unknowns"].append("Completion lengths and loss-bearing tokens require generated responses.")
    if method == "reward":
        result["unknowns"].append("Reward loss is per preference pair, not a causal token loss.")
    if result["errors"]:
        result["unknowns"].append("Aggregate token/preparation counts are unavailable because some rows could not be profiled.")
        for key in ("effective_rows", "raw_tokens", "retained_tokens", "discarded_tokens", "loss_tokens"):
            result[key] = None
    return result, examples, prompts, per_row


# Findings carry the complete row evidence, not a hidden first-N sample.
def _findings(profile, split):
    rules = (
        ("identity_errors", "warning", "measured", "Some source identities could not be established", "Use supported serializable content before relying on duplicate, overlap, or resume identity checks."),
        ("errors", "error", "projected", "Rows could not be profiled", "Correct the reported row or processor incompatibilities before training."),
        ("dropped_rows", "warning", "projected", "Preparation discards rows", "Review the affected rows and sequence budget; discarded rows do not train the model."),
        ("truncated_rows", "warning", "projected", "Sequence settings discard tokens", "Inspect the affected rows before increasing max_length or changing the truncation/packing strategy."),
        ("zero_loss_rows", "warning", "projected", "Rows have no loss-bearing tokens after the causal shift", "Review masks and sequence lengths; a single unmasked first token supplies no causal learning target."),
        ("duplicates", "warning", "measured", "Identical examples repeat", "Confirm repetition is intentional; duplicate rows increase those examples' training weight."),
        ("conflicting_labels", "warning", "measured", "Identical examples have conflicting preference labels", "Review contradictory labels before interpreting training progress."),
        ("identical_preference_rows", "warning", "measured", "Chosen and rejected responses are identical", "Remove or correct pairs that cannot teach a preference."),
        ("boundary_mismatch_rows", "warning", "projected", "Prompt tokenization changes at the response boundary", "Inspect prompt/response formatting; TRL slices by prompt length despite the mismatch."),
    )
    return [{"code": f"data.{split}.{key}", "severity": severity, "basis": basis,
             "summary": f"{split}: {summary}", "evidence": {"split": split, key: profile[key]},
             "recommendation": recommendation}
            for key, severity, basis, summary, recommendation in rules if profile[key]]


# Full scan consumes only loaded data and an already configured processor; no
# model loading, network calls, scorer imports or training preparation side effects.
def scan(cfg, processor, train_set, eval_set, *, primary_rows=None, progress=None):
    processor = copy.deepcopy(processor)
    template = None
    if cfg.method.name == "sft":
        tokenizer = getattr(processor, "tokenizer", processor)
        if cfg.args.eos_token is not None and not getattr(cfg.args, "chat_template_path", None):
            if cfg.args.eos_token not in tokenizer.get_vocab():
                raise ValueError(f"eos_token {cfg.args.eos_token!r} is absent from the tokenizer vocabulary")
            tokenizer.eos_token = cfg.args.eos_token
        if (cfg.args.assistant_only_loss and not chat_encoding.is_non_jinja(processor) and
                not has_generation_markers(processor.chat_template)):
            template = get_training_chat_template(processor)
    train_rows = list(train_set)
    if primary_rows is not None and not 0 <= primary_rows <= len(train_rows):
        raise ValueError("primary_rows must identify a prefix of the mixed training dataset")
    # SFT resolves automatic completion loss from TRAIN shape once; evaluation
    # must not independently choose a different loss when its shape differs.
    completion_only = getattr(cfg.args, "completion_only_loss", None)
    if completion_only is None:
        completion_only = bool(train_rows and "prompt" in train_rows[0])
    train, train_examples, train_prompts, contributions = _split(
        cfg, processor, train_rows, "train", progress, template, completion_only)
    result = {"version": 1, "method": cfg.method.name, "train": train, "eval": None,
              "overlap": {"identical_examples": [], "prompts": []}, "findings": _findings(train, "train")}
    if primary_rows is not None:
        train["primary_rows"] = primary_rows
        train["replay_rows"] = len(train_rows) - primary_rows
        # Provenance follows each token through packing, so mixed blocks still
        # attribute retained tokens and active causal targets to their sources.
        train["sources"] = {}
        for name, start, end in (("primary", 0, primary_rows), ("replay", primary_rows, len(train_rows))):
            subset = contributions[start:end]
            train["sources"][name] = {"rows": end - start,
                "raw_tokens": sum(row["raw_tokens"] for row in subset),
                "retained_tokens": sum(row["retained_tokens"] for row in subset),
                "loss_tokens": None if train["loss_tokens"] is None else
                    sum(row["loss_tokens"] for row in subset if row["loss_tokens"] is not None)}
            if train["errors"]:
                for key in ("raw_tokens", "retained_tokens", "loss_tokens"):
                    train["sources"][name][key] = None
    if eval_set is not None:
        evaluation, eval_examples, eval_prompts, _ = _split(
            cfg, processor, list(eval_set), "eval", progress, template, completion_only)
        result["eval"] = evaluation
        result["findings"].extend(_findings(evaluation, "eval"))
        result["overlap"] = _overlap(train_examples, train_prompts, eval_examples, eval_prompts)
        for name, overlap in result["overlap"].items():
            if overlap:
                result["findings"].append({"code": f"data.overlap.{name}", "severity": "warning", "basis": "measured",
                    "summary": ("Training and evaluation contain identical examples" if name == "identical_examples"
                                else "Training and evaluation reuse prompts; responses may differ"),
                    "evidence": {name: overlap}, "recommendation": "Use independently held-out examples when measuring generalization."})
    return result
