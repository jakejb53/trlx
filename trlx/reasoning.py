"""Map separate reasoning into the effective chat template and verify its loss-bearing tokens."""

import collections
import copy
import re

from datasets import Dataset
from transformers import ProcessorMixin
from trl import pack_dataset
from trl.chat_template_utils import add_response_schema, get_training_chat_template, has_generation_markers
from trl.data_utils import _tokenize, prepare_multimodal_messages

from dataset.progress import stage
from trlx import TrlxError, chat_encoding, data_profile


# Resolve metadata on a private processor: validation must not change the trainer's processor.
def _resolve(cfg, processor, progress):
    if getattr(cfg.args, "chat_template_path", None):
        from trlx import model

        # Reuse the existing override projector; the actual trainer still applies its own override.
        view = model.assessment_processor(cfg, progress=progress)
    else:
        view = copy.deepcopy(processor)
    tokenizer = getattr(view, "tokenizer", view)
    # Template cloning owns EOS after the configured EOS override, just as in SFTTrainer.
    if not getattr(cfg.args, "chat_template_path", None) and cfg.args.eos_token is not None:
        tokenizer.eos_token = cfg.args.eos_token
    if getattr(tokenizer, "response_template", None) is None:
        # This is an explicit library lookup by chat template, never a model-name guess.
        add_response_schema(view)
    schema = tokenizer.response_template
    fields = schema.get("fields", {}) if isinstance(schema, dict) else {}
    candidates = [key for key, value in fields.items() if key != "content" and
                  isinstance(value, dict) and value.get("content") == "text" and not value.get("repeats")]
    if len(candidates) != 1:
        raise ValueError("response_template must identify exactly one non-content text field for reasoning")
    template = None
    if (cfg.args.assistant_only_loss and not chat_encoding.is_non_jinja(view) and
            not has_generation_markers(view.chat_template)):
        template = get_training_chat_template(view)
    return view, template, candidates[0]


# Rendering must use the same message normalization as TRL's tokenization helper.
def _render(processor, messages, kwargs):
    if isinstance(processor, ProcessorMixin):
        messages = prepare_multimodal_messages(messages)
    try:
        text = processor.apply_chat_template(messages, tokenize=False, **kwargs)
    except Exception as error:
        # The template is operator/model-provided code; expose its failure with the dataset row context.
        raise ValueError(f"chat template rendering failed: {error}") from error
    if not isinstance(text, str):
        raise ValueError("chat template did not render a single text conversation")
    return text


# One separate trace has exactly one destination; never overwrite a different native trace.
def _map_row(row, field):
    if any(key in row for key in ("input_ids", "labels", "assistant_masks", "completion_mask", "prompt", "completion")):
        raise ValueError("requires raw messages rows, without prompt/completion or prepared token columns")
    reasoning = row.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("reasoning must be a nonempty string")
    messages = copy.deepcopy(row.get("messages"))
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
        raise ValueError("messages must be a nonempty conversation")
    assistants = [m for m in messages if m.get("role") == "assistant"]
    if len(assistants) != 1 or messages[-1].get("role") != "assistant":
        raise ValueError("a separate reasoning column requires exactly one assistant response at the end")
    if not all(isinstance(m.get("content"), str) for m in messages):
        raise ValueError("include-reasoning requires text message contents")
    assistant = assistants[0]
    if assistant.get(field) not in (None, "", reasoning):
        raise ValueError(f"reasoning conflicts with assistant.{field}")
    assistant[field] = reasoning
    return messages


# A field probe locates the actual instance even when identical text also occurs in the prompt.
def _field_span(messages, field, rendered, processor, kwargs, *, message_index=-1):
    probe = copy.deepcopy(messages)
    marker = "TRLX_REASONING_FIELD_PROBE"
    while marker in rendered:
        marker += "_"
    probe[message_index][field] = marker
    location = f"messages[{message_index % len(messages)}].{field}"
    sample = _render(processor, probe, kwargs)
    if sample.count(marker) != 1:
        raise ValueError(f"chat template does not render {location} exactly once")
    prefix, suffix = sample.split(marker)
    end = len(rendered) - len(suffix) if suffix else len(rendered)
    start = len(prefix)
    if not rendered.startswith(prefix) or not rendered.endswith(suffix):
        raise ValueError(f"chat template changes surrounding text with {location}; mapping is not verifiable")
    if rendered[start:end].strip() != messages[message_index][field].strip():
        raise ValueError(f"chat template discards or rewrites {location}")
    return start, end


# Fit the actual token sequence without re-rendering or guessing a characters-to-tokens ratio.
# Keep native structure and a causal predecessor; remaining capacity belongs to reasoning first.
def _fit_tokens(record, offsets, rendered, messages, body_span, processor, kwargs, maximum, reasoning_only):
    if maximum is None or len(record["input_ids"]) <= maximum:
        return record

    # Tokens straddling a field boundary are indivisible and remain with the template structure.
    def positions(span):
        start, end = span
        return {index for index, (left, right) in enumerate(offsets) if start <= left < right <= end}

    reasoning = positions(body_span)
    answer = positions(_field_span(messages, "content", rendered, processor, kwargs))
    context = set()
    for index in range(len(messages) - 1):
        context.update(positions(_field_span(messages, "content", rendered, processor, kwargs,
                                             message_index=index)))
    retained = set(range(len(record["input_ids"]))) - reasoning - answer - context
    retained.add(0)  # A first selected reasoning token must still have a preceding causal input.
    if maximum <= len(retained):
        raise ValueError(f"--max-length {maximum} cannot fit the required template structure and a reasoning token; "
                         f"use at least {len(retained) + 1}")
    budget = maximum - len(retained)
    # Final-answer content has no training role in reasoning-only mode. For ordinary inclusion,
    # it is the secondary target; user/system content receives only the remaining prefix budget.
    groups = (reasoning, context) if reasoning_only else (reasoning, answer, context)
    for group in groups:
        selected = sorted(group - retained)[:budget]
        retained.update(selected)
        budget -= len(selected)
    indices = sorted(retained)
    return {key: [values[index] for index in indices] for key, values in record.items()}


# Native metadata owns delimiters. A terminating reasoning boundary must be learnable, not guessed.
def _reasoning_bounds(rendered, start, end, definition):
    opening = re.escape(definition["open"]) if isinstance(definition.get("open"), str) else definition.get("open_pattern")
    closing = re.escape(definition["close"]) if isinstance(definition.get("close"), str) else definition.get("close_pattern")
    if not isinstance(opening, str) or not isinstance(closing, str):
        raise ValueError("reasoning-only loss requires explicit opening and closing boundaries in response_template")
    try:
        candidates = [match for match in re.finditer(opening, rendered)
                      if match.end() <= start and not rendered[match.end():start].strip()]
        suffix = rendered[end:]
        whitespace = len(suffix) - len(suffix.lstrip())
        close = re.match(closing, suffix[whitespace:])
    except re.error as error:
        raise ValueError(f"invalid reasoning boundary pattern: {error}") from error
    if len(candidates) != 1 or close is None or not close.group().strip():
        raise ValueError("reasoning boundaries are missing or ambiguous in the rendered assistant response")
    if not candidates[0].group().strip():
        # Parser lookaheads can rely on a separate start anchor; they do not identify opening tokens.
        raise ValueError("reasoning-only loss requires a nonempty native opening boundary")
    return candidates[0].start(), end + whitespace + close.end()


# Probe the field's exact rendered location, then verify real tokens and the selected loss mask.
def _validate_row(row, messages, field, processor, template, args, number, *, reasoning_only=False):
    kwargs = {"chat_template": template, **data_profile._template_kwargs(row)}
    rendered = _render(processor, messages, kwargs)
    start, end = _field_span(messages, field, rendered, processor, kwargs)
    body_span = (start, end)
    if reasoning_only:
        tokenizer = getattr(processor, "tokenizer", processor)
        start, end = _reasoning_bounds(rendered, start, end, tokenizer.response_template["fields"][field])
        answer_start, answer_end = _field_span(messages, "content", rendered, processor, kwargs)
        if max(start, answer_start) < min(end, answer_end):
            raise ValueError("reasoning boundary overlaps the final answer")
    # Offset mappings prove which tokens came from this field, even if its text also occurs in the question.
    if isinstance(processor, ProcessorMixin):
        processor_kwargs = copy.deepcopy(kwargs.pop("processor_kwargs", None) or {})
        processor_kwargs["return_offsets_mapping"] = True
        # ProcessorMixin consumes offsets when constructing assistant_masks. Retrieve them
        # separately through its supported API, then require identical tokens before combining.
        encoded = _tokenize(processor, messages, **kwargs, processor_kwargs=copy.deepcopy(processor_kwargs),
                            return_assistant_tokens_mask=False)
        if args.assistant_only_loss:
            masked = _tokenize(processor, messages, **kwargs, processor_kwargs=copy.deepcopy(processor_kwargs),
                               return_assistant_tokens_mask=True)
            if masked["input_ids"] != encoded["input_ids"]:
                raise ValueError("processor produced different tokens for offsets and assistant masks; "
                                 "cannot verify reasoning supervision")
            encoded["assistant_masks"] = masked.get("assistant_masks")
    else:
        tokenizer_kwargs = dict(kwargs.pop("tokenizer_kwargs", None) or {})
        tokenizer_kwargs["return_offsets_mapping"] = True
        encoded = _tokenize(processor, messages, **kwargs, tokenizer_kwargs=tokenizer_kwargs,
                            return_assistant_tokens_mask=args.assistant_only_loss)
    ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None or len(offsets) != len(ids):
        raise ValueError("tokenizer must supply offset mappings to verify reasoning supervision")
    mask = encoded.get("assistant_masks") if args.assistant_only_loss else [1] * len(ids)
    if mask is None or len(mask) != len(ids):
        raise ValueError("chat template does not supply the assistant loss mask")
    markers = [number if left < end and right > start else 0 for left, right in offsets]
    if reasoning_only and any(flag and (left < start or right > end)
                              for flag, (left, right) in zip(markers, offsets)):
        # A token is indivisible: never supervise part of an answer or prompt to include a delimiter.
        raise ValueError("a token crosses the reasoning-only loss boundary")
    covered = set()
    for flag, (left, right) in zip(markers, offsets):
        if flag:
            covered.update(range(max(start, left), min(end, right)))
    if not any(markers) or any(not rendered[i].isspace() and i not in covered for i in range(start, end)):
        raise ValueError("tokenization discards part of the reasoning")
    labels = [token if active and (not reasoning_only or flag) else -100
              for token, active, flag in zip(ids, mask, markers)]
    if any(flag and (index == 0 or labels[index] == -100) for index, flag in enumerate(markers)):
        raise ValueError("loss mask excludes reasoning tokens")
    record = {"input_ids": ids, "labels": labels, "_reasoning_rows": markers}
    return _fit_tokens(record, offsets, rendered, messages, body_span, processor,
                       {"chat_template": template, **data_profile._template_kwargs(row)},
                       args.max_length, reasoning_only)


# Verify that TRL packing/truncation preserves the targets selected by budget fitting.
def _validate_retention(records, args, split):
    expected = collections.Counter(source for record in records for source in record["_reasoning_rows"] if source)
    packing = args.packing
    if split == "eval" and args.eval_packing is not None:
        packing = args.eval_packing
    if packing:
        prepared = Dataset.from_list(records)
        if args.shuffle_dataset:
            prepared = prepared.shuffle(seed=args.seed, keep_in_memory=True)
        records = pack_dataset(prepared, args.max_length, args.packing_strategy,
                               {"num_proc": args.dataset_num_proc, "keep_in_memory": True})
    elif args.max_length is not None:
        window = slice(-args.max_length, None) if args.truncation_mode == "keep_end" else slice(None, args.max_length)
        records = [{key: value[window] for key, value in record.items()} for record in records]
    padding_free = args.padding_free or (args.packing and args.packing_strategy in {"bfd", "bfd_split"})
    actual = collections.Counter(record["_reasoning_rows"][position] for record in records
                                 for position in data_profile._loss_positions(record, padding_free)
                                 if record["_reasoning_rows"][position])
    for number, count in expected.items():
        if actual[number] != count:
            raise ValueError(f"{split} row {number}: truncation or packing excludes reasoning tokens "
                             f"({actual[number]}/{count} supervised); adjust max_length or packing")


# The same in-memory transformation owns primary, replay, and evaluation rows in every process.
def prepare(cfg, processor, train_set, eval_set, *, progress=None):
    if not getattr(cfg.dataset, "include_reasoning", False):
        return train_set, eval_set
    reasoning_only = getattr(cfg.dataset, "reasoning_only_loss", False)
    try:
        view, template, field = _resolve(cfg, processor, progress)
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
        raise TrlxError(f"--include-reasoning: cannot resolve reasoning format: {error}") from error
    results = []
    for split, dataset in (("train", train_set), ("eval", eval_set)):
        if dataset is None:
            results.append(None)
            continue
        records = []
        with stage(progress, f"including {split} reasoning", total=len(dataset), unit="rows") as activity:
            # Dataset.map keeps output in memory; source files and existing dataset caches remain untouched.
            def convert(row, index):
                try:
                    messages = _map_row(row, field)
                    tokens = _validate_row(row, messages, field, view, template, cfg.args, index + 1,
                                           reasoning_only=reasoning_only)
                    records.append(tokens)
                except (ValueError, TypeError, KeyError, AttributeError, NotImplementedError) as error:
                    raise TrlxError(f"--include-reasoning: {split} row {index + 1}: {error}") from error
                activity.advance()
                # Use the same fitted sequence in profiling and TRL; re-tokenizing messages would
                # restore discarded context. The original source fields remain unchanged for provenance.
                return {"messages": messages, "input_ids": tokens["input_ids"], "labels": tokens["labels"]}

            converted = dataset.map(convert, with_indices=True, keep_in_memory=True, load_from_cache_file=False)
            try:
                _validate_retention(records, cfg.args, split)
            except (ValueError, TypeError, KeyError) as error:
                raise TrlxError(f"--include-reasoning: {error}") from error
            results.append(converted)
    return tuple(results)
