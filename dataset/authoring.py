"""Shared request contracts and operations for browser and stateless CLI authoring."""

import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from dataset.endpoint import Endpoint
from dataset.io import DatasetError, append_examples


class Input(BaseModel):
    """Reject unknown fields and coercion at both authoring boundaries."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Sampling(Input):
    """Absent controls remain absent from the endpoint request."""
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=-1)
    max_tokens: int | None = Field(default=None, gt=0)
    presence_penalty: float | None = None
    repetition_penalty: float | None = Field(default=None, gt=0)


class Generation(Input):
    """One card is one independent endpoint request, including its credentials."""
    endpoint: str
    model: str = Field(min_length=1)
    api_key: str = ""
    user: str = Field(min_length=1)
    system: str = ""
    # Endpoint-owned message fields remain unrestricted, including tool calls and rich content.
    context: list[dict] = Field(default_factory=list)
    sampling: Sampling
    timeout: float = Field(gt=0)
    retries: int = Field(ge=0)


class Message(Input):
    """Saved examples contain text only, with role order validated by Example."""
    role: Literal["user", "assistant"]
    content: str


class Example(Input):
    """An omitted reasoning field differs from a present empty reasoning string."""
    messages: list[Message] = Field(min_length=2, max_length=2)
    reasoning: str | None = None

    # Never let a manually crafted save introduce system turns or null reasoning.
    @model_validator(mode="after")
    def validate_shape(self):
        if [message.role for message in self.messages] != ["user", "assistant"]:
            raise ValueError("messages must be ordered user, assistant")
        if "reasoning" in self.model_fields_set and self.reasoning is None:
            raise ValueError("reasoning must be text or omitted")
        return self


class Save(Input):
    """Only explicit save requests touch a training dataset."""
    path: str = Field(min_length=1)
    examples: list[Example] = Field(min_length=1)


# Validation messages expose field locations and constraints, never request values.
def validation_message(error):
    details = [f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in error.errors()]
    return "Invalid request: " + "; ".join(details)


# Convert expected CLI validation failures without retaining Pydantic's input-bearing cause.
def validate(model, value):
    try:
        return model.model_validate(value)
    except ValidationError as error:
        raise DatasetError(validation_message(error)) from None


# Python accepts non-JSON numeric constants by default; neither input boundary should emit them.
def _invalid_constant(value):
    raise ValueError("non-finite numeric constant")


# Numeric overflow must not silently turn Context data into an invalid JSON Infinity token.
def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds finite floating-point range")
    return number


# Report syntax positions without echoing prompts, context, or credentials from the document.
def read_json(stream, source):
    try:
        return json.load(stream, parse_constant=_invalid_constant, parse_float=_finite_float)
    except json.JSONDecodeError as error:
        raise DatasetError(f"{source}: invalid JSON at line {error.lineno}, column {error.colno}") from None
    except (OSError, UnicodeError, ValueError):
        raise DatasetError(f"{source}: cannot read UTF-8 JSON with finite representable numbers") from None


# One output has no persistent state; provider-owned context fields pass through unchanged.
def generate(body, *, progress=None):
    if not body.model.strip() or not body.user.strip():
        raise DatasetError("model and user prompt must not be blank")
    endpoint = Endpoint(body.endpoint, body.model, body.api_key or None, body.timeout, body.retries)
    messages = ([{"role": "system", "content": body.system}] if body.system else [])
    messages.extend(body.context)
    messages.append({"role": "user", "content": body.user})
    parameters = body.sampling.model_dump(exclude_none=True)
    max_tokens = parameters.pop("max_tokens", None)
    reply = endpoint.complete_full(messages, max_tokens, sampling=parameters, progress=progress)
    return {"answer": reply.content, "reasoning": reply.reasoning}


# Keep omitted reasoning distinct from an empty string when publishing selected examples.
def save(body, *, progress=None):
    examples = [example.model_dump(exclude_unset=True) for example in body.examples]
    return append_examples(body.path, examples, progress=progress)
