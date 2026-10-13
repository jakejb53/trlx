"""Report the waste labelled in one untouched generation and emit its removal edits.

Usage: python authoring/span_report.py PROMPT.txt RAW.json LABELS.json
                                       [--edits-out EDITS.json] [--config TOML]

RAW.json is the generator's stdout object with keys answer and reasoning, as
for edit.py. LABELS.json is a list of waste spans, one object each:

  field            "reasoning" or "answer"
  text             the exact waste text, of any size from a hedge word to a passage
  anchor           optional: a longer string containing text exactly once, used
                   when text alone is not unique in the field (hedge fragments)
  category         one of CATEGORIES below; "beyond-prompt" (a whole answer
                   section that serves none of the prompt's requests) is
                   accepted only on answer labels
  reason           one line
  retained         optional bool: kept because removal, even with a connective
                   repair, would leave the retained reasoning or answer incoherent
  retain_reason    required when retained is true
  error_introduced optional bool: the span withdraws a correct result

Validation fails unless every span (its anchor when given) occurs exactly once
in the untouched field, every text occurs exactly once in its anchor, and no
two spans overlap. Token counts use the scoring endpoint's echoed offsets over
the complete rendered example, so they are counted in context. The reasoning
budget is the token limit minus every rendered token outside the reasoning
span, after the answer's waste removals; a later answer correction moves it.

--edits-out writes the removals of non-retained spans in edit.py's format
(old = anchor or text, new = the same with the waste text deleted). Corrections
and connective repairs are appended to that list by the editor.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config  # noqa: E402
from count_score import post, render, spans  # noqa: E402

CATEGORIES = ("repetition", "abandoned-draft", "same-route-reverification",
              "rereading", "write-up-narration", "hedge", "beyond-prompt")
# A whole answer section that serves none of the prompt's requests is waste only in
# the answer; reasoning may explore beyond the request, so the category is rejected there.
ANSWER_ONLY = ("beyond-prompt",)


def locate(fields, labels):
    """Return [(field, start, end, waste_start, waste_end, label)] sorted by position.

    start/end delimit the edited span (anchor or text); waste_start/waste_end
    delimit the waste text inside it. Raises ValueError on any invalid label.
    """
    out = []
    for i, lab in enumerate(labels):
        where = f"label {i}"
        field = lab.get("field")
        if field not in fields:
            raise ValueError(f"{where}: field must be one of {sorted(fields)}")
        text = lab.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{where}: text must be a nonempty string")
        if lab.get("category") not in CATEGORIES:
            raise ValueError(f"{where}: category must be one of {CATEGORIES}")
        if lab["category"] in ANSWER_ONLY and field != "answer":
            raise ValueError(f"{where}: category {lab['category']!r} applies only to the answer")
        if not isinstance(lab.get("reason"), str) or not lab["reason"].strip():
            raise ValueError(f"{where}: reason is required")
        if lab.get("retained") and not str(lab.get("retain_reason", "")).strip():
            raise ValueError(f"{where}: retained spans require retain_reason")
        span = lab.get("anchor", text)
        if span is not text and span.count(text) != 1:
            raise ValueError(f"{where}: text occurs {span.count(text)} times in its anchor, expected 1")
        n = fields[field].count(span)
        if n != 1:
            raise ValueError(f"{where}: {'anchor' if span is not text else 'text'} occurs {n} times "
                             f"in the untouched {field}, expected 1: {span[:80]!r}")
        start = fields[field].index(span)
        ws = start + span.index(text)
        out.append((field, start, start + len(span), ws, ws + len(text), lab))
    out.sort(key=lambda r: (r[0], r[1]))
    for a, b in zip(out, out[1:]):
        if a[0] == b[0] and a[2] > b[1]:
            raise ValueError(f"spans overlap in {a[0]}: {a[5]['text'][:60]!r} and {b[5]['text'][:60]!r}")
    return out


def removal_edits(located):
    """edit.py entries deleting the waste text of every non-retained span, in position order."""
    edits = []
    for field, start, end, ws, we, lab in located:
        if lab.get("retained"):
            continue
        old = lab.get("anchor", lab["text"])
        new = old[: ws - start] + old[we - start:]
        edits.append({"field": field, "old": old, "new": new})
    return edits


def token_offsets(cfg, rendered):
    sc = cfg.scoring
    body = {"model": cfg.model, "prompt": rendered, "echo": bool(sc["echo"]),
            "max_tokens": int(sc["max_tokens"]), "logprobs": int(sc["logprobs"]),
            "add_special_tokens": bool(sc["add_special_tokens"])}
    return post(sc["url"], body)["choices"][0]["logprobs"]["text_offset"]


def count_in(offs, s, e):
    return sum(1 for o in offs if s <= o < e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("raw")
    ap.add_argument("labels")
    ap.add_argument("--edits-out")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = config.load(args.config)
    tok = cfg.tokenization
    raw = json.loads(pathlib.Path(args.raw).read_text(encoding="utf-8"))
    labels = json.loads(pathlib.Path(args.labels).read_text(encoding="utf-8"))
    # Labels address the untouched fields exactly as generated; offsets are
    # mapped into the rendering, which strips surrounding newlines.
    fields = {"reasoning": raw["reasoning"], "answer": raw["answer"]}
    try:
        located = locate(fields, labels)
    except ValueError as e:
        raise SystemExit(f"span_report: invalid labels: {e}")

    prompt = config.read_prompt(args.prompt)
    stripped = {k: v.strip("\n") for k, v in fields.items()}
    lead = {k: len(v) - len(v.lstrip("\n")) for k, v in fields.items()}
    count, rendered = render(cfg, prompt, stripped["reasoning"], stripped["answer"])
    sp = spans(rendered, stripped["reasoning"], stripped["answer"],
               tok["reasoning_open"], tok["reasoning_close"], tok["assistant_end"])
    offs = token_offsets(cfg, rendered)
    r0, r1 = sp["reasoning"]
    reasoning_tokens = count_in(offs, r0, r1)

    removed = {"reasoning": 0, "answer": 0}
    by_cat, flagged = {}, []
    print("waste spans (tokens, category, removed/retained, text):")
    for field, start, end, ws, we, lab in located:
        base = sp[field][0] - lead[field]
        n = count_in(offs, base + ws, base + we)
        state = "retained" if lab.get("retained") else "removed"
        if state == "removed":
            removed[field] += n
            by_cat[lab["category"]] = by_cat.get(lab["category"], 0) + n
        if lab.get("error_introduced"):
            flagged.append(lab)
        print(f"  {n:6}  {lab['category']:26} {state:8} {lab['text'][:70]!r}")
    print("removed by category:")
    for cat in CATEGORIES:
        if cat in by_cat:
            print(f"  {cat:26} {by_cat[cat]:6}")
    share = removed["reasoning"] / reasoning_tokens if reasoning_tokens else 0.0
    remaining = reasoning_tokens - removed["reasoning"]
    print(f"reasoning tokens: {reasoning_tokens}; waste removed: {removed['reasoning']} ({share:.0%}); "
          f"remaining: {remaining}")
    print(f"answer waste removed: {removed['answer']} tokens")
    # Answer compression happens before fitting, so the budget counts the answer
    # after its waste removals; later answer corrections still move it.
    outside = count - reasoning_tokens - removed["answer"]
    budget = cfg.limit - outside
    print(f"reasoning budget: {budget} (limit {cfg.limit} minus {outside} rendered tokens "
          f"outside the reasoning, after answer waste removal); remaining reasoning "
          f"{'FITS' if remaining <= budget else 'EXCEEDS the budget by ' + str(remaining - budget)}")
    for lab in flagged:
        print(f"ERROR INTRODUCED: {lab['text'][:100]!r}: {lab['reason']}")
    if args.edits_out:
        edits = removal_edits(located)
        pathlib.Path(args.edits_out).write_text(json.dumps(edits, indent=1, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
        print(f"wrote {args.edits_out}: {len(edits)} removal edit(s)")


if __name__ == "__main__":
    main()
