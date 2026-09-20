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
import math
import time
import urllib.error
import urllib.parse
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
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise DatasetError("retries must be an integer 0 or more")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or timeout <= 0 or (isinstance(timeout, float) and not math.isfinite(timeout))):
            raise DatasetError("timeout must be finite positive seconds")
        if not isinstance(url, str):
            raise DatasetError("endpoint URL must be a string containing an HTTP(S) base URL")
        try:
            parsed = urllib.parse.urlsplit(url)
            _ = parsed.port  # Access validates a supplied numeric port and its range.
        except ValueError:
            raise DatasetError("endpoint URL is invalid; supply an HTTP(S) base URL with a valid host and port")
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url)):
            raise DatasetError("endpoint URL is invalid; supply an HTTP(S) base URL without whitespace")
        # Header validation must never include the credential value in its error.
        if api_key is not None and not isinstance(api_key, str):
            raise DatasetError("API key must be text; correct the credential environment variable")
        if api_key is not None and any(c in api_key for c in "\r\n\0"):
            raise DatasetError("API key contains a line break or NUL; correct the credential environment variable")
        self.url = url.rstrip("/") + "/chat/completions"
        # Diagnostics omit URL credentials/query data; request construction is unchanged.
        self.display_url = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path.rstrip("/") + "/chat/completions", "", "")
        )
        self._base_url = url
        self._base_display_url = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", "")
        )
        self._secrets = sorted({value for value in (
            api_key, parsed.username, parsed.password,
            *(value for _, value in urllib.parse.parse_qsl(parsed.query)),
        ) if value}, key=len, reverse=True)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries

    # External error text may echo credentials; redact before truncating a response body.
    def _diagnostic(self, value):
        text = str(value).replace(self.url, self.display_url).replace(self._base_url, self._base_display_url)
        for secret in self._secrets:
            text = text.replace(secret, "[redacted]")
        return text

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
            try:
                req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return self._extract(json.loads(resp.read().decode("utf-8")))
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code not in _RETRY_STATUSES:
                    # Reading an error body may itself fail; the HTTP status stays final.
                    try:
                        detail = self._diagnostic(e.read().decode("utf-8", "replace"))[:500]
                    except (OSError, http.client.HTTPException) as read_error:
                        raise DatasetError(
                            f"{self.display_url}: {last}; cannot read error response: "
                            f"{self._diagnostic(read_error)}; check the endpoint response and connection"
                        ) from read_error
                    raise DatasetError(f"{self.display_url}: {last}: {detail}")
            except urllib.error.URLError as e:
                last = f"connection error: {self._diagnostic(e.reason)}"
            except TimeoutError:
                last = f"timed out after {self.timeout}s"
            # A connection that drops after the status line surfaces as an
            # http.client exception or a bare OSError, not a URLError. Both are
            # transient and retried like a refused connection.
            except (http.client.HTTPException, OSError) as e:
                last = f"connection dropped: {type(e).__name__}: {self._diagnostic(e)}"
            except json.JSONDecodeError:
                raise DatasetError(f"{self.display_url}: reply is not JSON; check the endpoint's chat-completions response")
            except UnicodeError:
                raise DatasetError(f"{self.display_url}: request or reply has invalid text encoding; check the endpoint and credentials")
            except ValueError:
                raise DatasetError(f"{self.display_url}: invalid request; check the endpoint URL and credential environment variable")
        raise DatasetError(
            f"{self.display_url}: gave up after {self.retries + 1} attempts; last error: {last}; "
            "check endpoint availability and the timeout/retries settings"
        )

    # Splits a chat completions reply into content and reasoning. `reasoning`
    # and `reasoning_content` are the two spellings servers use for the same
    # field; a null value means the model produced none.
    def _extract(self, reply):
        try:
            message = reply["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError):
            raise DatasetError(f"{self.display_url}: reply has no choices[0].message.content")
        if not isinstance(content, str):
            raise DatasetError(
                f"{self.display_url}: choices[0].message.content must be a string; check the endpoint response format"
            )
        for field in ("reasoning", "reasoning_content"):
            if message.get(field) is not None and not isinstance(message[field], str):
                raise DatasetError(
                    f"{self.display_url}: choices[0].message.{field} must be a string or null; "
                    "check the endpoint response format"
                )
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        return Reply(content, reasoning)

    # Assistant texts for many message lists, in request order.
    def complete_many(self, message_lists, concurrency, max_tokens=None):
        return [r.content for r in self.complete_many_full(message_lists, concurrency, max_tokens)]

    # Runs complete_full() over many message lists with `concurrency` threads.
    # Results are in request order. The first failure cancels pending requests
    # and is raised.
    def complete_many_full(self, message_lists, concurrency, max_tokens=None):
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise DatasetError("concurrency must be an integer at least 1")
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(self.complete_full, m, max_tokens) for m in message_lists]
            try:
                return [f.result() for f in futures]
            except DatasetError:
                for f in futures:
                    f.cancel()
                raise
