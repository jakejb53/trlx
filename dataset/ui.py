"""FastAPI boundary for browser-owned authoring state and explicit dataset saves."""

import pathlib
import sys

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from dataset import authoring
from dataset.authoring import Generation, Save
from dataset.failures import capture, render
from dataset.io import DatasetError
from dataset.progress import Progress


# Construct an application without reading browser state, datasets, or credentials.
def create_app():
    app = FastAPI(title="Dataset authoring")
    assets = pathlib.Path(__file__).with_name("ui")
    app.mount("/assets", StaticFiles(directory=assets), name="assets")

    # Never echo Pydantic's input payload: it may contain an API key or prompt.
    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        return JSONResponse({"error": authoring.validation_message(error)}, status_code=422)

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
        with Progress("dataset ui generation") as progress:
            return authoring.generate(body, progress=progress)

    # A retry after a lost HTTP response rechecks the authoritative file and skips
    # rows already published, so the browser can safely retain its pending copy.
    @app.post("/api/save")
    def save(body: Save):
        with Progress("dataset ui save") as progress:
            return authoring.save(body, progress=progress)

    return app


# Uvicorn owns HTTP serving and signal handling; this process owns no model or GPU.
def run(host, port):
    import uvicorn

    if not host.strip() or not 1 <= port <= 65535:
        raise DatasetError("--host must be nonempty and --port must be in 1..65535")
    uvicorn.run(create_app(), host=host, port=port)
    return 0
