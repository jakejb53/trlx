"""Continued-pretraining chunks: a text file into `text` rows.

Also home of the token estimate shared with stats and chat. No tokenizer is
loaded for estimates; the heuristic is a single documented constant.
"""

import math
import re

from dataset.io import DatasetError

# Characters per token assumed by the estimate. Prose on modern tokenizers runs
# near 4, so 3.5 overestimates slightly, which is the safe direction for a
# chunk limit. Label any displayed estimate with ESTIMATE_LABEL.
CHARS_PER_TOKEN = 3.5
ESTIMATE_LABEL = f"est. chars/{CHARS_PER_TOKEN:g}"

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
# Sentence end: terminal punctuation followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# Conservative token estimate: rounds up so a limit check never under-counts.
def estimate_tokens(text):
    return math.ceil(len(text) / CHARS_PER_TOKEN)


# Splits one over-long piece into pieces each within the limit. Tries sentence
# ends first; a single sentence over the limit is cut at the character count
# the limit implies.
def _split_oversized(text, max_tokens):
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    pieces = []
    for sentence in _SENTENCE_END.split(text):
        if estimate_tokens(sentence) <= max_tokens:
            pieces.append(sentence)
        else:
            pieces.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
    return pieces


# Greedily packs pieces into chunks joined by `sep`, each within the limit.
def _pack(pieces, max_tokens, sep):
    chunks, current = [], []
    for piece in pieces:
        candidate = sep.join(current + [piece])
        if current and estimate_tokens(candidate) > max_tokens:
            chunks.append(sep.join(current))
            current = [piece]
        else:
            current.append(piece)
    if current:
        chunks.append(sep.join(current))
    return chunks


# Paragraph-aware chunking. Paragraphs pack whole while they fit. One over the
# limit is split by sentence and pre-packed with spaces; the resulting pieces
# then join the paragraph-level pack, so a trailing remainder of a split
# paragraph may share a chunk with the next paragraph.
def chunk_text(text, max_tokens):
    if max_tokens < 1:
        raise DatasetError(f"--max-tokens must be at least 1, got {max_tokens}")
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(text) if p.strip()]
    pieces = []
    for p in paragraphs:
        if estimate_tokens(p) <= max_tokens:
            pieces.append(p)
        else:
            pieces.extend(_pack(_split_oversized(p, max_tokens), max_tokens, " "))
    return _pack(pieces, max_tokens, "\n\n")


# Reads a text file and returns one {"text": chunk} row per chunk.
def cpt_rows(path, max_tokens):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise DatasetError(f"{path}: cannot read: {e.strerror or e}; check the path and permissions")
    except UnicodeError:
        raise DatasetError(f"{path}: input is not valid UTF-8; convert the text to UTF-8")
    return [{"text": chunk} for chunk in chunk_text(text, max_tokens)]
