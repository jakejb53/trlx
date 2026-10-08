"""Apply exact literal edits to an untouched generation in one transformation.

Usage: python authoring/edit.py RAW.json EDITS.json OUT_DIR
       python authoring/edit.py derive RAW.json FINAL_DIR EDITS.json

RAW.json is the generator's stdout object with keys answer and reasoning.
EDITS.json is a list of {"field": "reasoning"|"answer", "old": str, "new": str}.
Every "old" span must occur exactly once in the untouched field; edits apply in
list order to a copy of the untouched text, never to an intermediate candidate.
Leading whitespace is removed from the answer. Writes OUT_DIR/reasoning.txt and
OUT_DIR/answer.txt and prints a unified diff per field.

derive goes the other way: FINAL_DIR holds the intended reasoning.txt and
answer.txt (edited copies of the untouched text), and the edit list that turns
the untouched text into them is written to EDITS.json. Each span is widened
until it is unique in the untouched field, overlapping spans are merged, and the
run fails unless applying the list reproduces FINAL_DIR exactly.
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


def derive_edits(raw, final):
    """Return the edit list that turns raw into final ({"reasoning", "answer"}).

    Change blocks come from a character-level diff. Each block is widened
    symmetrically until its untouched span is unique; blocks whose widened
    spans overlap are merged into one edit, since apply_edits replaces whole
    spans. Leading whitespace of the answer is not an edit: apply_edits strips it.
    """
    edits = []
    for field in ("reasoning", "answer"):
        src, dst = raw[field], final[field]
        if field == "answer":
            dst = dst.lstrip()
            src = src.lstrip()
        blocks = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2
                  in difflib.SequenceMatcher(None, src, dst, autojunk=False).get_opcodes()
                  if tag != "equal"]
        while True:
            spans = []
            for i1, i2, j1, j2 in blocks:
                k = 0
                while src.count(src[max(0, i1 - k):i2 + k]) != 1:
                    k += 1
                    assert i1 - k >= 0 or i2 + k <= len(src), f"{field}: no unique span for block {i1}:{i2}"
                spans.append((max(0, i1 - k), min(len(src), i2 + k)))
            merged = False
            for n in range(len(blocks) - 1):
                if spans[n][1] > spans[n + 1][0]:
                    a, b = blocks[n], blocks[n + 1]
                    blocks[n:n + 2] = [(a[0], b[1], a[2], b[3])]
                    merged = True
                    break
            if not merged:
                break
        for (i1, i2, j1, j2), (lo, hi) in zip(blocks, spans):
            edits.append({"field": field, "old": src[lo:hi], "new": src[lo:i1] + dst[j1:j2] + src[i2:hi]})
    applied = apply_edits(raw, edits)
    for field in ("reasoning", "answer"):
        want = final[field].lstrip() if field == "answer" else final[field]
        assert applied[field] == want, f"{field}: derived edits do not reproduce the final text"
    return edits


def derive_main(raw_path, final_dir, out_path):
    raw = json.load(open(raw_path, encoding="utf-8"))
    final_dir = pathlib.Path(final_dir)
    final = {f: (final_dir / f"{f}.txt").read_text(encoding="utf-8") for f in ("reasoning", "answer")}
    edits = derive_edits(raw, final)
    pathlib.Path(out_path).write_text(json.dumps(edits, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}: {len(edits)} edit(s); applying them reproduces {final_dir} exactly")


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "derive":
        derive_main(sys.argv[2], sys.argv[3], sys.argv[4])
        return
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
