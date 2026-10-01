"""FastAPI boundary for browser-owned authoring state and explicit dataset saves."""

import pathlib
import sys
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataset.endpoint import Endpoint
from dataset.failures import capture, render
from dataset.io import DatasetError, append_examples
from dataset.progress import Progress


class Input(BaseModel):
    """Reject unknown fields and coercion at the browser boundary."""
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


# Construct an application without reading browser state, datasets, or credentials.
def create_app():
    app = FastAPI(title="Dataset authoring")
    assets = pathlib.Path(__file__).with_name("ui")
    app.mount("/assets", StaticFiles(directory=assets), name="assets")

    # Never echo Pydantic's input payload: it may contain an API key or prompt.
    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        details = [f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in error.errors()]
        return JSONResponse({"error": "Invalid request: " + "; ".join(details)}, status_code=422)

    # Reuse credential-safe boundary diagnostics without exposing request objects.
    @app.exception_handler(DatasetError)
    async def expected_failure(request: Request, error: DatasetError):
        return JSONResponse({"error": render(capture(error))}, status_code=400)

    # Unexpected failures remain diagnosable on stderr; browser state remains intact.
    @app.exception_handler(Exception)
    async def unexpected_failure(request: Request, error: Exception):
        report = capture(error)
        print(render(report, include_traceback=True), file=sys.stderr, flush=True)
        return JSONResponse({"error": render(report)}, status_code=500)

    # The page and API share an origin, including when the operator uses a proxy.
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(assets / "index.html", headers={"Cache-Control": "no-cache"})

    # Synchronous handlers use FastAPI's worker pool; slow endpoints do not block
    # other cards or saves. Nothing from a request becomes backend persistent state.
    @app.post("/api/generate")
    def generate(body: Generation):
        if not body.model.strip() or not body.user.strip():
            raise DatasetError("model and user prompt must not be blank")
        endpoint = Endpoint(body.endpoint, body.model, body.api_key or None, body.timeout, body.retries)
        messages = ([{"role": "system", "content": body.system}] if body.system else [])
        messages.extend(body.context)
        messages.append({"role": "user", "content": body.user})
        parameters = body.sampling.model_dump(exclude_none=True)
        max_tokens = parameters.pop("max_tokens", None)
        with Progress("dataset ui generation") as progress:
            reply = endpoint.complete_full(messages, max_tokens, sampling=parameters, progress=progress)
        return {"answer": reply.content, "reasoning": reply.reasoning}

    # A retry after a lost HTTP response rechecks the authoritative file and skips
    # rows already published, so the browser can safely retain its pending copy.
    @app.post("/api/save")
    def save(body: Save):
        examples = [example.model_dump(exclude_unset=True) for example in body.examples]
        with Progress("dataset ui save") as progress:
            return append_examples(body.path, examples, progress=progress)

    return app


# Uvicorn owns HTTP serving and signal handling; this process owns no model or GPU.
def run(host, port):
    import uvicorn

    if not host.strip() or not 1 <= port <= 65535:
        raise DatasetError("--host must be nonempty and --port must be in 1..65535")
    uvicorn.run(create_app(), host=host, port=port)
    return 0
