"""Token length distribution per column; with --model, per-token log-prob.

Without --model, lengths are the cpt heuristic estimate and are labelled so.
With --model, the tokenizer loaded for scoring gives exact counts. No
tokenizer is ever loaded only for counting.
"""

import statistics

from dataset.convert import final_assistant_index
from dataset.cpt import ESTIMATE_LABEL, estimate_tokens
from dataset.io import DatasetError
from dataset.progress import stage

# Columns scored for log-prob when present. Each is conditioned on the row's
# `prompt` column; `messages` is conditioned on its own preceding turns.
RESPONSE_COLUMNS = ("completion", "chosen", "rejected")


# True for a list of {"role", "content"} dicts, the shape of every
# conversational column (messages, prompt, chosen, rejected, completion).
def _is_message_list(value):
    return isinstance(value, list) and bool(value) and all(
        isinstance(m, dict) and "role" in m and "content" in m for m in value
    )


# Columns to report: every column holding a string or a message list in the
# first row, or the explicit --columns list validated against the first row.
def _columns(rows, requested):
    if not rows:
        raise DatasetError("input has no rows")
    if requested:
        for c in requested:
            if c not in rows[0]:
                raise DatasetError(f"--columns {c}: first row has no column '{c}'")
        return list(requested)
    return [k for k, v in rows[0].items() if isinstance(v, str) or _is_message_list(v)]


# Renders a message list as text for estimation: contents joined by newlines.
def _messages_text(value):
    return "\n".join(str(m.get("content", "")) for m in value if isinstance(m, dict))


# Summary statistics for a list of numbers, keyed by label.
def _summary(values):
    values = sorted(values)
    return {
        "count": len(values),
        "min": values[0],
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": values[min(len(values) - 1, int(0.9 * len(values)))],
        "max": values[-1],
    }


# Length distribution per column. `count` maps text to a token count.
def lengths(rows, columns, count, *, progress=None):
    out = {}
    for column in columns:
        values = []
        with stage(progress, f"counting tokens in {column}", total=len(rows), unit="rows") as activity:
            for i, row in enumerate(rows):
                if column not in row:
                    raise DatasetError(f"row {i} has no column '{column}'")
                v = row[column]
                text = _messages_text(v) if isinstance(v, list) else v
                if not isinstance(text, str):
                    raise DatasetError(f"row {i} column '{column}' is {type(v).__name__}, expected text")
                values.append(count(text))
                activity.advance()
            out[column] = _summary(values)
    return out


# Prints one table of summaries. `unit` names what the numbers are.
def print_table(title, table, unit):
    print(f"{title} ({unit})")
    keys = ["count", "min", "mean", "median", "p90", "max"]
    print(f"{'column':<20}" + "".join(f"{k:>10}" for k in keys))
    for column, s in table.items():
        print(f"{column:<20}" + "".join(f"{s[k]:>10.1f}" if isinstance(s[k], float) else f"{s[k]:>10}" for k in keys))


# Loads model and tokenizer once. Imported lazily so plain stats never pays
# for torch and transformers.
def _load(model_path):
    import torch
    import transformers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
        model.to(device).eval()
    except torch.cuda.OutOfMemoryError as e:
        raise DatasetError(
            f"--model {model_path}: out of memory loading model on {device}; "
            "free device memory or choose a smaller model"
        ) from e
    except (OSError, ValueError) as e:
        raise DatasetError(
            f"--model {model_path}: cannot load: {e}; check the model path/ID, access, and model files"
        )
    print(f"model loaded on {device}")
    return tokenizer, model, device


# Token ids for (context, context + response). Strings concatenate; message
# lists go through the chat template, with the generation prompt appended to
# the context so the response tokens start where the model would generate.
def _encode_pair(tokenizer, context, response, index):
    if isinstance(context, str) and isinstance(response, str):
        ctx = tokenizer(context, add_special_tokens=True)["input_ids"]
        full = tokenizer(context + response, add_special_tokens=True)["input_ids"]
    elif isinstance(context, list) and isinstance(response, list):
        from jinja2 import TemplateError

        # return_dict=False asks for a plain id list on every transformers
        # version; the default return type changed between major versions.
        # These failures belong to the supplied conversation/template boundary.
        try:
            ctx = tokenizer.apply_chat_template(
                context, add_generation_prompt=True, tokenize=True, return_dict=False
            )
            full = tokenizer.apply_chat_template(context + response, tokenize=True, return_dict=False)
        except (TemplateError, ValueError, TypeError, IndexError, KeyError) as e:
            raise DatasetError(
                f"row {index}: cannot apply the model chat template: {e}; "
                "check the messages and the model's chat template"
            )
    else:
        raise DatasetError(
            f"row {index}: prompt is {type(context).__name__} and response is "
            f"{type(response).__name__}; both must be strings or both lists"
        )
    if full[: len(ctx)] != ctx:
        raise DatasetError(f"row {index}: tokenizer does not extend the prompt tokens into the response")
    return ctx, full


# Mean per-token log-prob of the response tokens under the model, or None
# when there are no scoreable response tokens. logp[k] is the log-prob of
# token k+1, so response tokens start at index len(ctx)-1. With an empty
# context the first token has nothing predicting it and is skipped.
def _logprob(model, device, ctx, full):
    import torch

    ids = torch.tensor([full], device=device)
    with torch.no_grad():
        logits = model(ids).logits[0, :-1].float()
    targets = ids[0, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None])[:, 0]
    response = logp[max(len(ctx), 1) - 1 :]
    if response.numel() == 0:
        return None
    return response.mean().item()


# The (context, response) pair for one row and one response column. Message
# validation is convert.final_assistant_index, the single owner of that rule.
def _response_pair(row, column, index):
    if column == "messages":
        last = final_assistant_index(row["messages"], index)
        return row["messages"][:last], row["messages"][last:]
    if column not in row:
        raise DatasetError(f"row {index} has no column '{column}'")
    if "prompt" not in row:
        raise DatasetError(f"row {index}: column '{column}' needs a 'prompt' column to condition on")
    return row["prompt"], row[column]


# Per-column mean log-prob summaries. Rows are scored one at a time. Rows
# with no scoreable response tokens are counted and reported, never dropped
# silently, so the summary count can be reconciled with the row count.
def logprobs(rows, tokenizer, model, device, *, progress=None):
    import torch

    present = [c for c in RESPONSE_COLUMNS + ("messages",) if c in rows[0]]
    out = {}
    for column in present:
        values = []
        empty = 0
        with stage(progress, f"scoring {column}", total=len(rows), unit="rows") as activity:
            for i, row in enumerate(rows):
                if "messages" not in row and column == "messages":
                    raise DatasetError(f"row {i} has no column 'messages'")
                ctx, full = _encode_pair(tokenizer, *_response_pair(row, column, i), i)
                # Resource exhaustion is an operator failure; other model bugs still propagate.
                try:
                    lp = _logprob(model, device, ctx, full)
                except torch.cuda.OutOfMemoryError as e:
                    raise DatasetError(
                        f"row {i}, column '{column}': out of memory scoring {len(full)} tokens on {device}; "
                        "free device memory, shorten the input, or choose a smaller model"
                    ) from e
                if lp is None:
                    empty += 1
                else:
                    values.append(lp)
                # Counts inspected rows, including empty responses that cannot be scored.
                activity.advance()
            if empty:
                print(f"{column}: {empty} rows with an empty response were not scored")
            if values:
                out[column] = _summary(values)
    return out


# Entry point for the subcommand. Prints tables; returns nothing.
def run(rows, columns=None, model_path=None, *, progress=None):
    columns = _columns(rows, columns)
    if model_path is None:
        print_table("token lengths", lengths(rows, columns, estimate_tokens, progress=progress), ESTIMATE_LABEL)
        return
    with stage(progress, f"loading model and tokenizer {model_path}"):
        tokenizer, model, device = _load(model_path)
    count = lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])
    print_table("token lengths", lengths(rows, columns, count, progress=progress), f"exact, {model_path}")
    print()
    print_table("mean per-token log-prob of response", logprobs(rows, tokenizer, model, device, progress=progress), "nats")
