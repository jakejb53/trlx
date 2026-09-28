"""OpenAI-compatible chat completions client over urllib.

Shared by dataset chat and, per the import rule, by trlx (llm_judge,
replay-build). Concurrency is a thread pool; order of results matches order
of requests. Retries cover connection errors, timeouts, retryable HTTP statuses,
and malformed response encoding/JSON/fields with exponential backoff.
Other HTTP statuses and explicit incomplete-generation results are final errors.
"""

import collections
import concurrent.futures
import http.client
import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from dataset.io import DatasetError
from dataset.progress import stage
from dataset import failures

# A reply's two parts. `reasoning` is the endpoint's own reasoning field and is
# "" when the endpoint returns none: either the model is not a reasoning model,
# or the server was started without a reasoning parser and left the reasoning
# inline in `content` (dataset chat rejects that case).
Reply = collections.namedtuple("Reply", "content reasoning")

_RETRY_STATUSES = {429, 500, 502, 503, 504}
# Deliberate constant, not a flag: an accepted exception to the no-runtime-
# defaults principle, recorded in PLAN.md. Attempt k waits base * 2**(k-1).
_BACKOFF_BASE_SECONDS = 1.0


# Batch-local request clocks never imply token generation or provider-side progress.
class _BatchRequests:
    # The reporter's clock permits deterministic timing tests and matches stage elapsed time.
    def __init__(self, total, attempts, timeout, clock):
        self.states = [{"state": "queued", "started": None, "attempt_started": None, "attempt": 0}
                       for _ in range(total)]
        self.attempts, self.timeout, self.clock = attempts, timeout, clock
        self.lock = threading.Lock()

    # Update under a separate lock; release it before any call into the reporter.
    def update(self, index, state, attempt=None):
        with self.lock:
            item = self.states[index]
            now = self.clock()
            item["state"] = state
            if attempt is not None:
                item["attempt"] = attempt
            if state == "awaiting endpoint":
                if item["started"] is None:
                    item["started"] = now
                item["attempt_started"] = now

    # Report every active request, including a final straggler, without exposing request content.
    def describe(self):
        with self.lock:
            now = self.clock()
            completed = sum(item["state"] == "completed" for item in self.states)
            queued = sum(item["state"] == "queued" for item in self.states)
            parts = [f"{completed} completed responses retained in memory", f"{queued} queued"]
            for index, item in enumerate(self.states):
                if item["state"] not in ("awaiting endpoint", "retry backoff"):
                    continue
                parts.append(f"request {index + 1}: {item['state']}, elapsed {now - item['started']:.1f}s, "
                             f"attempt {item['attempt']}/{self.attempts}" +
                             (f", attempt elapsed {now - item['attempt_started']:.1f}s"
                              if item["state"] == "awaiting endpoint" else ""))
            parts.append(f"socket timeout {self.timeout:g}s (not a total request deadline); server progress unavailable")
            return "; ".join(parts)


# One URL/credential policy serves ordinary messages and complete failure reports.
def diagnostic_redactor(url, api_key=None):
    parsed = urllib.parse.urlsplit(url)
    display = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
    values = {value for value in (
        api_key, parsed.username, parsed.password,
        *(value for _, value in urllib.parse.parse_qsl(parsed.query)),
        *(part.partition("=")[2] for part in parsed.query.split("&")),
        parsed.fragment,
    ) if value}
    # Servers may echo encoded, decoded, or form-encoded credentials separately from the URL.
    values.update(urllib.parse.unquote(value) for value in tuple(values))
    values.update(urllib.parse.quote(value, safe="") for value in tuple(values))
    values.update(urllib.parse.quote_plus(value, safe="") for value in tuple(values))
    secrets = sorted(values, key=len, reverse=True)

    # This closure stays local; failure transport contains only its sanitized output.
    def sanitize(value):
        text = str(value).replace(url, display)
        for secret in secrets:
            text = text.replace(secret, "[redacted]")
        return text

    return sanitize


class Endpoint:
    # url is the API base, e.g. http://host:8000/v1; /chat/completions is appended.
    # api_key None sends no Authorization header. timeout is per request in
    # seconds; retries is the number of re-attempts after the first failure.
    def __init__(self, url, model, api_key, timeout, retries):
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise DatasetError("retries (--retries in dataset commands) must be an integer 0 or more; use 0 to disable retries "
                               "or a positive count of additional attempts after the first request")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or timeout <= 0 or (isinstance(timeout, float) and not math.isfinite(timeout))):
            raise DatasetError("timeout (--timeout in dataset commands) must be finite positive seconds; use e.g. 120, "
                               "not 0, a negative number, or infinity")
        if not isinstance(url, str):
            raise DatasetError("endpoint URL must be a string containing an HTTP(S) base URL")
        try:
            parsed = urllib.parse.urlsplit(url)
            _ = parsed.port  # Access validates a supplied numeric port and its range.
        except ValueError:
            # Malformed URLs cannot reliably be decomposed for credential redaction.
            raise DatasetError("endpoint URL is invalid; supply an HTTP(S) base URL with a valid host and port") from None
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
        self._redact = diagnostic_redactor(url, api_key)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries

    # External error text may echo credentials; redact before truncating a response body.
    def _diagnostic(self, value):
        return self._redact(value)

    # One chat completion. Returns the assistant text; callers that need the
    # reasoning field call complete_full.
    def complete(self, messages, max_tokens=None, *, progress=None):
        with stage(progress, f"request to {self.display_url}") as activity:
            return self.complete_full(messages, max_tokens, progress=activity).content

    # One chat completion as a Reply. Raises DatasetError with the URL and last
    # failure when retries are exhausted. Strict callers
    # require explicit completion metadata; legacy callers retain their behavior.
    def complete_full(self, messages, max_tokens=None, *, progress=None, request="request", require_stop=False,
                      _status=None, sampling=None):
        try:
            return self._complete_full(messages, max_tokens, progress=progress, request=request,
                                       require_stop=require_stop, _status=_status, sampling=sampling)
        except Exception as error:
            # Preserve causes and traceback, applying the boundary's redaction to the whole report.
            failures.redact(error, self._diagnostic)
            failures.annotate(error, context={"endpoint": self.display_url, "operation": "endpoint request"})
            raise

    # Execute the existing retry policy beneath the single diagnostic-redaction boundary.
    def _complete_full(self, messages, max_tokens=None, *, progress=None, request="request", require_stop=False,
                       _status=None, sampling=None):
        body = {"model": self.model, "messages": messages}
        # Only explicit overrides travel to the server; connection and message fields
        # cannot be replaced through sampling. Provider-specific support stays explicit.
        if sampling is not None:
            allowed = {"temperature", "top_p", "top_k", "presence_penalty", "repetition_penalty"}
            if not isinstance(sampling, dict) or set(sampling) - allowed:
                raise DatasetError("sampling contains unsupported parameter names")
            for name, value in sampling.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise DatasetError(f"sampling {name} must be a finite number")
            body.update(sampling)
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = None
        # The final attempt remains the cause; the outer boundary redacts its full chain.
        last_error = None
        for attempt in range(self.retries + 1):
            if attempt:
                delay = _BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
                if _status is not None:
                    _status("retry backoff", attempt + 1)
                if progress is not None:
                    progress.note(f"{request}: {last}; retry attempt {attempt + 1}/{self.retries + 1} in {delay:g}s")
                time.sleep(delay)
            if _status is not None:
                _status("awaiting endpoint", attempt + 1)
            try:
                req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    # Retry only response decoding/schema failures here. Request errors and
                    # explicit generation limits below must not become transient failures.
                    try:
                        raw = json.loads(resp.read().decode("utf-8"))
                        reply = self._extract(raw)
                    except json.JSONDecodeError as error:
                        last_error = error
                        last = "reply is not JSON; check the endpoint's chat-completions response"
                        continue
                    except UnicodeDecodeError as error:
                        last_error = error
                        last = "reply has invalid UTF-8 encoding; check the endpoint response"
                        continue
                    except DatasetError as error:
                        last_error = error
                        last = self._diagnostic(error)
                        continue
                    if require_stop:
                        reason = raw["choices"][0].get("finish_reason")
                        if reason != "stop":
                            detail = self._diagnostic(repr(reason))[:200]
                            remedy = ("increase --max-tokens" if reason == "length" else
                                      "check the endpoint's completion metadata and response")
                            raise DatasetError(
                                f"{self.display_url}: expected finish_reason='stop', got {detail}; {remedy}"
                            )
                    return reply
            except urllib.error.HTTPError as e:
                last_error = e
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
                last_error = e
                last = f"connection error: {self._diagnostic(e.reason)}"
            except TimeoutError as error:
                last_error = error
                last = f"timed out after {self.timeout}s"
            # A connection that drops after the status line surfaces as an
            # http.client exception or a bare OSError, not a URLError. Both are
            # transient and retried like a refused connection.
            except (http.client.HTTPException, OSError) as e:
                last_error = e
                last = f"connection dropped: {type(e).__name__}: {self._diagnostic(e)}"
            except UnicodeError:
                raise DatasetError(f"{self.display_url}: request or reply has invalid text encoding; check the endpoint and credentials")
            except ValueError:
                raise DatasetError(f"{self.display_url}: invalid request; check the endpoint URL and credential environment variable")
        raise DatasetError(
            f"{self.display_url}: gave up after {self.retries + 1} attempts; last error: {last}; "
            "check endpoint availability and the timeout/retries settings",
            context={"attempts": self.retries + 1},
        ) from last_error

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
    def complete_many(self, message_lists, concurrency, max_tokens=None, *, progress=None, label="requests"):
        return [r.content for r in self.complete_many_full(
            message_lists, concurrency, max_tokens, progress=progress, label=label)]

    # Runs complete_full() over many message lists with `concurrency` threads.
    # Observe completion order but store input order. Report failure before the
    # pool waits for running requests; queued requests are cancelled immediately.
    # on_complete(index, reply) runs serially in the collecting thread, once per
    # successful request; callback failures use the same batch cancellation path.
    def complete_many_full(self, message_lists, concurrency, max_tokens=None, *, progress=None,
                           label="requests", require_stop=False, on_complete=None):
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise DatasetError("concurrency (--concurrency in dataset commands) must be an integer at least 1; use 1 "
                               "for sequential requests or a larger number for simultaneous requests")
        message_lists = list(message_lists)
        with stage(progress, f"{label} from {self.model} at {self.display_url}",
                   total=len(message_lists), unit="requests") as activity:
            tracker = _BatchRequests(len(message_lists), self.retries + 1, self.timeout,
                                     activity.reporter.clock if activity.reporter is not None else time.monotonic)
            activity.waiting_detail = tracker.describe

            # Each worker owns its request's state; only the collector advances measured batch counts.
            def complete(index, messages):
                # Subclasses/mocks that do not report attempts still have an accurate request lifetime.
                tracker.update(index, "awaiting endpoint", 1)
                # Never hold the tracking lock across endpoint work or progress emission.
                def status(state, attempt):
                    tracker.update(index, state, attempt)

                try:
                    reply = self.complete_full(messages, max_tokens, progress=activity,
                        request=f"request {index + 1}", _status=status,
                        **({"require_stop": True} if require_stop else {}))
                except BaseException:
                    tracker.update(index, "failed")
                    raise
                tracker.update(index, "completed")
                return reply

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(complete, i, messages): i
                           for i, messages in enumerate(message_lists)}
                replies = [None] * len(futures)
                try:
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            replies[futures[future]] = future.result()
                            if on_complete is not None:
                                on_complete(futures[future], replies[futures[future]])
                        except DatasetError as error:
                            # Request order is source-row order for eval-build.
                            raise DatasetError(
                                f"{label}: request {futures[future] + 1}: {self._diagnostic(error)}"
                            ) from error
                        activity.advance()
                except BaseException as error:
                    for pending in futures:
                        if pending.cancel():
                            tracker.update(futures[pending], "cancelled")
                    # Error diagnostics already sanitize endpoint responses. Sanitize
                    # again at this boundary before publishing a batch-level notice.
                    detail = self._diagnostic(error) if isinstance(error, DatasetError) else type(error).__name__
                    running = sum(f.running() for f in futures)
                    activity.note(f"stopping batch: {detail}; waiting for {running} in-flight request(s)")
                    raise
                return replies
