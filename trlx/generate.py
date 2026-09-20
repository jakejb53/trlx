"""Local generation shared by verify and replay-build.

Greedy decoding, one prompt at a time: the callers pass a few dozen prompts
and want reproducible text, not throughput. A prompt is a string or a
`messages` list; strings become a single user turn when the tokenizer has a
chat template, and are fed raw otherwise.
"""

import torch

from dataset.progress import stage
from trlx import TrlxError, data_load

# Tokens generated per prompt. Enough for a behaviour difference to show; a
# display-scale constant, the same accepted exception as POLL_MS.
MAX_NEW_TOKENS = 64

# Prompts used when the config names none ([verify].prompts absent, or
# `trlx verify` without --prompts). Generic by design: they name no domain,
# so any instruction-tuned model answers them and any fine-tune moves them.
BUILTIN_PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a haiku about a river in winter.",
    "What is 17 multiplied by 23? Show the steps.",
    "List three differences between a list and a tuple in Python.",
    "Summarise the plot of a story about a lighthouse keeper in one paragraph.",
    "Give a polite one-line reply declining a meeting invitation.",
    "Translate 'good morning, how are you?' into French and Spanish.",
    "Name the planets of the solar system in order from the sun.",
]


# Prompts for a config.DatasetRef, or the built-in set for None. A `prompt`
# column is taken as is (string or messages); a `messages` column is cut
# after its last user turn so the model has something to answer.
def prompts_from(ref, *, progress=None):
    if ref is None:
        return list(BUILTIN_PROMPTS)
    dataset = data_load.load_ref(ref, progress=progress)
    if "prompt" in dataset.column_names:
        with stage(progress, "preparing generation prompts", total=dataset.num_rows, unit="prompts") as activity:
            prompts = list(dataset["prompt"])
            activity.advance(len(prompts))
            return prompts
    if "messages" in dataset.column_names:
        prompts = []
        with stage(progress, "preparing generation prompts", total=dataset.num_rows, unit="prompts") as activity:
            for row, messages in enumerate(dataset["messages"], 1):
                if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
                    raise TrlxError(f"{ref.source}: row {row}: messages must be a list of message objects")
                last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=None)
                if last_user is None:
                    raise TrlxError(f"{ref.source}: row {row}: messages has no user turn to prompt with")
                prompts.append(messages[: last_user + 1])
                activity.advance()
        return prompts
    raise TrlxError(
        f"{ref.source}: prompts need a 'prompt' or 'messages' column; columns: {', '.join(dataset.column_names)}"
    )


# Chat-templated text for one prompt, and whether a template was applied.
# A templated string already carries its special tokens.
def _render(tokenizer, prompt):
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True), True
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), True
    return prompt, False


# Greedy completions for `prompts`, decoded without special tokens. The
# model is put in eval mode for the duration and restored after.
def generate(model, processor, prompts, max_new_tokens=MAX_NEW_TOKENS, *, progress=None):
    tokenizer = getattr(processor, "tokenizer", processor)
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    was_training = model.training
    model.eval()
    outputs = []
    try:
        with stage(progress, "generating completions", total=len(prompts), unit="prompts") as activity, torch.no_grad():
            for prompt in prompts:
                text, templated = _render(tokenizer, prompt)
                encoded = tokenizer(text, return_tensors="pt", add_special_tokens=not templated).to(device)
                generated = model.generate(
                    **encoded, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id
                )
                outputs.append(tokenizer.decode(generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True))
                # Count completed, decoded outputs, never merely submitted prompts.
                activity.advance()
    except torch.cuda.OutOfMemoryError as e:
        raise TrlxError("CUDA memory exhausted during generation; free GPU memory, shorten prompts, "
                        "or reduce the completion token limit") from e
    finally:
        model.train(was_training)
    return outputs
