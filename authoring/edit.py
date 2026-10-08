"""Apply exact literal edits to an untouched generation in one transformation.

Usage: python authoring/edit.py RAW.json EDITS.json OUT_DIR

RAW.json is the generator's stdout object with keys answer and reasoning.
EDITS.json is a list of {"field": "reasoning"|"answer", "old": str, "new": str}.
Every "old" span must occur exactly once in the untouched field; edits apply in
list order to a copy of the untouched text, never to an intermediate candidate.
Leading whitespace is removed from the answer. Writes OUT_DIR/reasoning.txt and
OUT_DIR/answer.txt and prints a unified diff per field.
"""
import difflib
import json
import pathlib
import sys


def apply_edits(raw, edits):
    """Return {"reasoning": str, "answer": str} with the edits applied."""
    assert set(raw) >= {"answer", "reasoning"}, f"unexpected response keys: {list(raw)}"
    text = {"reasoning": raw["reasoning"], "answer": raw["answer"]}
    for i, e in enumerate(edits):
        f = e["field"]
        n = text[f].count(e["old"])
        assert n == 1, f"edit {i} ({f}): span occurs {n} times, expected 1: {e['old'][:80]!r}"
        text[f] = text[f].replace(e["old"], e["new"])
    text["answer"] = text["answer"].lstrip()
    return text


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    raw = json.load(open(sys.argv[1], encoding="utf-8"))
    edits = json.load(open(sys.argv[2], encoding="utf-8"))
    out = pathlib.Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    text = apply_edits(raw, edits)
    for f in ("reasoning", "answer"):
        (out / f"{f}.txt").write_text(text[f], encoding="utf-8")
        sys.stdout.writelines(difflib.unified_diff(
            raw[f].splitlines(keepends=True), text[f].splitlines(keepends=True),
            fromfile=f"raw/{f}", tofile=f"edited/{f}", n=1))
        print()
    print(f"reasoning: {len(raw['reasoning'])} -> {len(text['reasoning'])} chars; "
          f"answer: {len(raw['answer'])} -> {len(text['answer'])} chars")


if __name__ == "__main__":
    main()
