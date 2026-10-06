"""Model-metadata-selected chat encoders that implement the Transformers tokenizer contract."""

import contextlib
import copy
import types
from collections.abc import Callable

from transformers import PreTrainedTokenizerBase
from transformers.utils.chat_template_utils import get_json_schema

from trlx import deepseek_v4_encoding


DEEPSEEK_V4_ARCHITECTURE = "DeepseekV4ForCausalLM"
_ADAPTER_ATTRIBUTE = "_trlx_chat_encoding"
_ORIGINAL_RESPONSE_ATTRIBUTE = "_trlx_original_response_template"
# TRL inspects this string during SFT construction even though rendering is owned by Python.
_TRAINER_CAPABILITY_TEMPLATE = "{% generation %}{% endgeneration %}"


# Transformers response parsing uses the same native boundaries as the bundled encoder.
DEEPSEEK_V4_RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": deepseek_v4_encoding.ASSISTANT_SP_TOKEN,
    "fields": {
        "reasoning": {
            "open": deepseek_v4_encoding.thinking_start_token,
            "close": deepseek_v4_encoding.thinking_end_token,
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": (
                r'(?:\n\n)?(?:<｜DSML｜tool_calls>\s*)?'
                r'<｜DSML｜invoke name="(?P<name>[^"]+)">\s*'
            ),
            "close_pattern": r"\s*</｜DSML｜invoke>\s*(?:</｜DSML｜tool_calls>)?",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": (
                    r'<｜DSML｜parameter name="(?P<key>[^"]+)" string="(?:true|false)">'
                    r"(?P<value>.*?)</｜DSML｜parameter>"
                ),
                "value_parser": {"name": "json", "args": {"allow_non_json": True}},
            },
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
        "content": {"close": deepseek_v4_encoding.eos_token, "content": "text"},
    },
}


# Convert callable tools exactly as Transformers' Jinja renderer does before handing schemas to DeepSeek.
def _normalise_tools(tools):
    if tools is None:
        return None
    result = []
    for tool in tools:
        if isinstance(tool, dict):
            result.append(tool)
        elif isinstance(tool, Callable):
            result.append(get_json_schema(tool))
        else:
            raise TypeError("tools must contain OpenAI function schemas or typed callables")
    return result or None


# Request-level tools belong on the first system message, matching vLLM's maintained renderer.
def _attach_tools(messages, tools):
    messages = copy.deepcopy(messages)
    if not tools:
        return messages
    index = next((index for index, message in enumerate(messages) if message.get("role") == "system"), None)
    if index is None:
        messages.insert(0, {"role": "system", "content": "", "tools": tools})
    else:
        messages[index]["tools"] = tools
    return messages


# DeepSeek's renderer owns thinking mode; absent request controls use its canonical no-prefix mode.
def _render_options(kwargs):
    thinking = kwargs.get("thinking")
    enabled = kwargs.get("enable_thinking")
    thinking_enabled = bool(thinking) or bool(enabled)
    if "thinking" not in kwargs and "enable_thinking" not in kwargs:
        thinking_enabled = True
    effort = kwargs.get("reasoning_effort")
    if effort == "none":
        thinking_enabled = False
        effort = None
    elif effort in (None, "low", "minimal", "medium"):
        effort = "low" if thinking_enabled else None
    elif effort not in ("high", "max"):
        raise ValueError("reasoning_effort must be none, low, minimal, medium, high, or max")
    return {
        "thinking_mode": "thinking" if thinking_enabled else "chat",
        "drop_thinking": kwargs.get("drop_thinking", True),
        "reasoning_effort": effort,
    }


# Continue-final-message removes only the final EOS; DeepSeek has no separate Jinja continuation phase.
def _prepare_conversation(conversation, tools, continue_final_message, add_generation_prompt):
    if not isinstance(conversation, (list, tuple)) or not conversation:
        raise ValueError("cannot encode an empty conversation")
    if not all(isinstance(message, dict) for message in conversation):
        raise TypeError("conversation messages must be dictionaries")
    messages = _attach_tools(conversation, tools)
    if continue_final_message:
        if add_generation_prompt:
            raise ValueError("continue_final_message and add_generation_prompt are incompatible")
        if continue_final_message not in (True, "content"):
            raise ValueError("DeepSeek-V4 can continue only the final assistant content field")
        if messages[-1].get("role") != "assistant":
            raise ValueError("continue_final_message requires a final assistant message")
        messages[-1]["wo_eos"] = True
    return messages


# Render one already-normalized conversation through the project-owned vLLM-derived encoder.
def _render(messages, options):
    return deepseek_v4_encoding.encode_messages(messages, **options)


# Locate one rendered field without trusting its text to be unique elsewhere in the conversation.
def _field_span(messages, index, field, rendered, options):
    probe = copy.deepcopy(messages)
    marker = "TRLX_DEEPSEEK_V4_FIELD_PROBE"
    while marker in rendered:
        marker += "_"
    probe[index][field] = marker
    sample = _render(probe, options)
    if sample.count(marker) != 1:
        return None
    prefix, suffix = sample.split(marker)
    if not rendered.startswith(prefix) or not rendered.endswith(suffix):
        raise ValueError(f"DeepSeek-V4 encoding changes surrounding text for messages[{index}].{field}")
    return len(prefix), len(rendered) - len(suffix), suffix


# Assistant masks cover the native reasoning opener through DSML calls and the EOS token.
def _assistant_spans(messages, rendered, options):
    spans = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        reasoning = _field_span(messages, index, "reasoning", rendered, options)
        content = _field_span(messages, index, "content", rendered, options)
        if content is None:
            raise ValueError(f"DeepSeek-V4 encoding does not render messages[{index}].content")
        start = reasoning[0] if reasoning is not None else content[0]
        if reasoning is not None and rendered[:start].endswith(deepseek_v4_encoding.thinking_start_token):
            start -= len(deepseek_v4_encoding.thinking_start_token)
        content_end, suffix = content[1], content[2]
        terminator = suffix.find(deepseek_v4_encoding.eos_token)
        if terminator < 0:
            if index != len(messages) - 1 or not message.get("wo_eos"):
                raise ValueError(f"DeepSeek-V4 assistant message {index} has no end-of-sentence token")
            end = len(rendered)
        else:
            end = content_end + terminator + len(deepseek_v4_encoding.eos_token)
        spans.append((start, end))
    return spans


# Map rendered character spans through the fast tokenizer after its own truncation and padding decisions.
def _assistant_masks(encoded, spans_by_row, batched, return_tensors):
    rows = encoded["input_ids"] if batched or return_tensors else [encoded["input_ids"]]
    masks = []
    for row, spans in enumerate(spans_by_row):
        mask = [0] * len(rows[row])
        for start, end in spans:
            first = encoded.char_to_token(row, start)
            last = encoded.char_to_token(row, end - 1)
            if first is None:
                continue
            stop = last + 1 if last is not None else len(mask)
            for position in range(first, stop):
                mask[position] = 1
        masks.append(mask)
    if not batched and not return_tensors:
        return masks[0]
    return masks


# Implement Transformers' tokenizer API while delegating explicit Jinja overrides to Transformers itself.
def _apply_deepseek_v4_template(
    self,
    conversation,
    tools=None,
    documents=None,
    chat_template=None,
    add_generation_prompt=False,
    continue_final_message=False,
    tokenize=True,
    padding=False,
    truncation=False,
    max_length=None,
    return_tensors=None,
    return_dict=True,
    return_assistant_tokens_mask=False,
    tokenizer_kwargs=None,
    **kwargs,
):
    if chat_template not in (None, _TRAINER_CAPABILITY_TEMPLATE):
        return PreTrainedTokenizerBase.apply_chat_template(
            self,
            conversation,
            tools=tools,
            documents=documents,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )
    if documents:
        raise ValueError("DeepSeek-V4 encoding does not support chat-template documents")
    if return_assistant_tokens_mask and not (tokenize and return_dict):
        raise ValueError("return_assistant_tokens_mask requires tokenize=True and return_dict=True")
    if isinstance(conversation, (list, tuple)) and conversation and isinstance(conversation[0], (list, tuple)):
        conversations, batched = conversation, True
    else:
        conversations, batched = [conversation], False
    schemas = _normalise_tools(tools)
    options = _render_options(kwargs)
    prepared = [
        _prepare_conversation(item, schemas, continue_final_message, add_generation_prompt)
        for item in conversations
    ]
    rendered = [_render(item, options) for item in prepared]
    spans = ([_assistant_spans(item, text, options) for item, text in zip(prepared, rendered, strict=True)]
             if return_assistant_tokens_mask else None)
    if not tokenize:
        return rendered if batched else rendered[0]
    call_kwargs = dict(tokenizer_kwargs or {})
    call_kwargs.update({"padding": padding, "truncation": truncation, "add_special_tokens": False})
    if max_length is not None:
        call_kwargs["max_length"] = max_length
    if return_tensors is not None:
        call_kwargs["return_tensors"] = return_tensors
    encoded = self(rendered if batched else rendered[0], **call_kwargs)
    if not return_dict:
        return encoded["input_ids"]
    if return_assistant_tokens_mask:
        # The conditional construction above guarantees spans exist only for this branch.
        encoded["assistant_masks"] = _assistant_masks(encoded, spans, batched, return_tensors)
        if return_tensors:
            encoded.convert_to_tensors(tensor_type=return_tensors)
    return encoded


# The architecture registry is the only place model metadata selects a non-Jinja implementation.
def adapt_processor(processor, model_config):
    architectures = getattr(model_config, "architectures", None) or []
    if DEEPSEEK_V4_ARCHITECTURE not in architectures:
        return processor
    if not isinstance(processor, PreTrainedTokenizerBase):
        raise TypeError("DeepSeek-V4 chat encoding requires a Transformers tokenizer processor")
    if getattr(processor, _ADAPTER_ATTRIBUTE, None) == "deepseek_v4":
        return processor
    setattr(processor, _ADAPTER_ATTRIBUTE, "deepseek_v4")
    setattr(processor, _ORIGINAL_RESPONSE_ATTRIBUTE, getattr(processor, "response_template", None))
    processor.response_template = copy.deepcopy(DEEPSEEK_V4_RESPONSE_TEMPLATE)
    processor.apply_chat_template = types.MethodType(_apply_deepseek_v4_template, processor)
    return processor


# Explicit Jinja ownership removes the instance method and restores the model's original parser metadata.
def remove_adapter(processor):
    if getattr(processor, _ADAPTER_ATTRIBUTE, None) is None:
        return processor
    if "apply_chat_template" in processor.__dict__:
        del processor.__dict__["apply_chat_template"]
    processor.response_template = getattr(processor, _ORIGINAL_RESPONSE_ATTRIBUTE, None)
    delattr(processor, _ADAPTER_ATTRIBUTE)
    delattr(processor, _ORIGINAL_RESPONSE_ATTRIBUTE)
    return processor


# An explicit template overrides automatic encoding while leaving the adapter available for later runs.
def is_non_jinja(processor):
    return (getattr(processor, _ADAPTER_ATTRIBUTE, None) is not None and
            getattr(processor, "chat_template", None) is None)


# TRL inspects chat_template only while constructing SFT; the capability marker must never reach checkpoints.
@contextlib.contextmanager
def trainer_capabilities(processor):
    if not is_non_jinja(processor) or getattr(processor, "chat_template", None) is not None:
        yield
        return
    processor.chat_template = _TRAINER_CAPABILITY_TEMPLATE
    try:
        yield
    finally:
        processor.chat_template = None
