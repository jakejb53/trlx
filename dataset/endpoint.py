"""OpenAI-compatible chat completions client over urllib.

Shared by dataset chat and, per the import rule, by trlx (llm_judge,
replay-build). Concurrency is a thread pool; order of results matches order
of requests. Retries cover connection errors, timeouts, 429, and 5xx with
exponential backoff. Any other HTTP status is a final error.
"""

import collections
import concurrent.futures
import http.client
import json
import time
import urllib.error
import urllib.request

from dataset.io import DatasetError

# A reply's two parts. `reasoning` is the endpoint's own reasoning field and is
# "" when the endpoint returns none: either the model is not a reasoning model,
# or the server was started without a reasoning parser and left the reasoning
# inline in `content` (dataset chat rejects that case).
Reply = collections.namedtuple("Reply", "content reasoning")

_RETRY_STATUSES = {429, 500, 502, 503, 504}
# Deliberate constant, not a flag: an accepted exception to the no-runtime-
# defaults principle, recorded in PLAN.md. Attempt k waits base * 2**(k-1).
_BACKOFF_BASE_SECONDS = 1.0


class Endpoint:
    # url is the API base, e.g. http://host:8000/v1; /chat/completions is appended.
    # api_key None sends no Authorization header. timeout is per request in
    # seconds; retries is the number of re-attempts after the first failure.
    def __init__(self, url, model, api_key, timeout, retries):
        if retries < 0:
            raise DatasetError(f"retries must be 0 or more, got {retries}")
        if timeout <= 0:
            raise DatasetError(f"timeout must be positive seconds, got {timeout}")
        self.url = url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries

    # One chat completion. Returns the assistant text; callers that need the
    # reasoning field call complete_full.
    def complete(self, messages, max_tokens=None):
        return self.complete_full(messages, max_tokens).content

    # One chat completion as a Reply. Raises DatasetError with the URL and last
    # status when retries are exhausted or the reply is malformed.
    def complete_full(self, messages, max_tokens=None):
        body = {"model": self.model, "messages": messages}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
            req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return self._extract(json.loads(resp.read().decode("utf-8")))
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code not in _RETRY_STATUSES:
                    raise DatasetError(f"{self.url}: {last}: {e.read().decode('utf-8', 'replace')[:500]}")
            except urllib.error.URLError as e:
                last = f"connection error: {e.reason}"
            except TimeoutError:
                last = f"timed out after {self.timeout}s"
            # A connection that drops after the status line surfaces as an
            # http.client exception or a bare OSError, not a URLError. Both are
            # transient and retried like a refused connection.
            except (http.client.HTTPException, OSError) as e:
                last = f"connection dropped: {type(e).__name__}: {e}"
            except json.JSONDecodeError:
                raise DatasetError(f"{self.url}: reply is not JSON")
        raise DatasetError(f"{self.url}: gave up after {self.retries + 1} attempts; last error: {last}")

    # Splits a chat completions reply into content and reasoning. `reasoning`
    # and `reasoning_content` are the two spellings servers use for the same
    # field; a null value means the model produced none.
    def _extract(self, reply):
        try:
            message = reply["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError):
            raise DatasetError(f"{self.url}: reply has no choices[0].message.content")
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        return Reply(content, reasoning)

    # Assistant texts for many message lists, in request order.
    def complete_many(self, message_lists, concurrency, max_tokens=None):
        return [r.content for r in self.complete_many_full(message_lists, concurrency, max_tokens)]

    # Runs complete_full() over many message lists with `concurrency` threads.
    # Results are in request order. The first failure cancels pending requests
    # and is raised.
    def complete_many_full(self, message_lists, concurrency, max_tokens=None):
        if concurrency < 1:
            raise DatasetError(f"concurrency must be at least 1, got {concurrency}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(self.complete_full, m, max_tokens) for m in message_lists]
            try:
                return [f.result() for f in futures]
            except DatasetError:
                for f in futures:
                    f.cancel()
                raise
