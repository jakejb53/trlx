"""Text file to a messages dataset via an OpenAI-compatible endpoint.

Pass 1: each chunk plus the questions instruction yields up to N questions.
Pass 2: each question plus its chunk yields the answer. Each pass runs over
all its requests concurrently; pass 2 starts after pass 1 completes.

Rows carry the answer's reasoning in a `reasoning` column beside `messages`,
taken from the endpoint's reasoning field and "" when it returns none. The
endpoint must return reasoning in that field, which means a reasoning parser
on a self-hosted server; a reply with the reasoning inline is fatal unless
--strip-reasoning-tags removes it (see strip_inline_reasoning).
"""

import re

from dataset.cpt import chunk_text
from dataset.io import DatasetError
from dataset.progress import stage
from dataset.prompts import fill

# Leading list markers a model tends to add: "1.", "1)", "-", "*".
_LIST_MARKER = re.compile(r"^\s*(\d+[.)]|[-*])\s*")
# Any XML-style tag opening a reply, matched by shape rather than by name so
# no model's reasoning convention is written into this source (SPEC line 7).
_INLINE_TAG = re.compile(r"^\s*<([A-Za-z][\w-]*)>")


# Reasoning belongs in the endpoint's reasoning field, not in the content: a
# reply opening with an XML-style tag means the server is returning it inline,
# which would otherwise be parsed as questions or written into an answer. Fatal
# by default because one misconfigured endpoint spoils every reply, not one
# row. strip removes the block instead, for an endpoint the operator cannot
# reconfigure. A block that never closes was truncated by the model's token
# limit; there is nothing to strip and the remainder is not an answer, so that
# is fatal either way. `where` names the chunk or question in the message.
def strip_inline_reasoning(content, where, strip):
    match = _INLINE_TAG.match(content)
    if not match:
        return content
    tag = match.group(1)
    close = f"</{tag}>"
    end = content.find(close)
    if end < 0:
        raise DatasetError(
            f"{where}: reply opens a <{tag}> block that never closes; the reply was cut by the "
            f"model's token limit"
        )
    if not strip:
        raise DatasetError(
            f"{where}: reply begins with a <{tag}> block, so the endpoint is returning reasoning "
            f"inline. Start the server with a reasoning parser so reasoning is returned in the "
            f"response field, or pass --strip-reasoning-tags to remove the block."
        )
    return content[end + len(close):].lstrip()


# One question per non-empty line, list markers stripped, at most n.
def parse_questions(reply, n):
    questions = []
    for line in reply.splitlines():
        line = _LIST_MARKER.sub("", line).strip()
        if line:
            questions.append(line)
    return questions[:n]


# Runs both passes. Returns (rows, skipped) where skipped lists messages for
# chunks that yielded no questions and questions that yielded empty answers.
def build(text, max_tokens, n, questions_endpoint, answers_endpoint, questions_prompt,
          answers_prompt, concurrency, strip_reasoning_tags=False, *, progress=None):
    with stage(progress, "chunking source text") as activity:
        chunks = chunk_text(text, max_tokens, progress=activity)
        activity.note(f"{len(chunks)} source chunks; requesting up to {n} questions per chunk")
    if not chunks:
        raise DatasetError("input text is empty")
    skipped = []

    requests = [[{"role": "user", "content": fill(questions_prompt, n=n, chunk=c)}] for c in chunks]
    replies = questions_endpoint.complete_many_full(requests, concurrency, progress=progress, label="questions")
    pairs = []  # (chunk index, question)
    for i, reply in enumerate(replies):
        # Pass 1 discards reasoning: its output is question strings, not data.
        content = strip_inline_reasoning(reply.content, f"chunk {i}", strip_reasoning_tags)
        questions = parse_questions(content, n)
        if not questions:
            skipped.append(f"chunk {i}: no questions parsed from reply")
        pairs.extend((i, q) for q in questions)

    requests = [
        [{"role": "user", "content": fill(answers_prompt, chunk=chunks[i], question=q)}]
        for i, q in pairs
    ]
    replies = answers_endpoint.complete_many_full(requests, concurrency, progress=progress, label="answers")
    rows = []
    for (i, q), reply in zip(pairs, replies):
        where = f"chunk {i}, question: {q}"
        answer = strip_inline_reasoning(reply.content, where, strip_reasoning_tags).strip()
        if not answer:
            skipped.append(f"chunk {i}: empty answer for question: {q}")
            continue
        # The answer's reasoning is kept in its own column rather than folded
        # into the assistant turn: the endpoint returns it separately, and
        # wrapping it back into the content would invent a format the model
        # never emitted. "" when the endpoint returned none.
        rows.append({
            "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": answer}],
            "reasoning": reply.reasoning,
        })
    return rows, skipped
