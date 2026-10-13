"""Report how editing moved a row's likelihood, as shifts only.

Usage: python authoring/shift_report.py PROMPT.txt SAVED_REASONING.txt|- SAVED_ANSWER.txt
                                        EDITED_REASONING.txt|- EDITED_ANSWER.txt [--config TOML]

Scores three renderings under the saved probes with count_score.py's render,
spans, and score, so masks and boundaries cannot differ from a single-example
measurement: the saved row, the candidate, and the saved reasoning with the
edited answer (the fixed-context comparison that isolates answer wording).

Prints the candidate's rendered token count against the full-sequence limit
and, for the reasoning, answer, and combined spans, the shift in mean
log-probability, candidate minus saved row, with the scored token counts on
both sides. Absolute means are never printed: the shift is the quantity
ROW-REPAIR.md step 7 reports, and printing absolutes invites reporting the
wrong figure. A shift whose span is absent on either side is n/a. When the
edited reasoning equals the saved reasoning, the third scoring is skipped
because it would equal the candidate.

Exit status: 0; 2 when the candidate exceeds the limit (shifts are still
printed so the size problem and the edits can be judged together).
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config  # noqa: E402
import count_score  # noqa: E402


def measure(cfg, prompt, reasoning, answer):
    """Rendered token count and {span: (mean_logprob, scored_tokens)} for one example."""
    tok = cfg.tokenization
    count, rendered = count_score.render(cfg, prompt, reasoning, answer)
    sp = count_score.spans(rendered, reasoning, answer,
                           tok["reasoning_open"], tok["reasoning_close"], tok["assistant_end"])
    res, _ = count_score.score(cfg, rendered, sp)
    return count, res


def shift_line(label, saved, candidate, name):
    if name not in saved or name not in candidate:
        return f"shift[{label}]: n/a"
    (sm, sn), (cm, cn) = saved[name], candidate[name]
    return f"shift[{label}]: {cm - sm:+.4f} ({sn} -> {cn} tokens)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("saved_reasoning", help="saved row's reasoning file, or - for an answer-only row")
    ap.add_argument("saved_answer")
    ap.add_argument("edited_reasoning", help="candidate's reasoning file, or - for an answer-only row")
    ap.add_argument("edited_answer")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = config.load(args.config)
    prompt = config.read_prompt(args.prompt)
    read = lambda a: None if a == "-" else config.read_field(a)  # noqa: E731
    saved_r, saved_a = read(args.saved_reasoning), config.read_field(args.saved_answer)
    edited_r, edited_a = read(args.edited_reasoning), config.read_field(args.edited_answer)

    _, saved = measure(cfg, prompt, saved_r, saved_a)
    count, candidate = measure(cfg, prompt, edited_r, edited_a)
    limit = cfg.limit
    print(f"candidate_rendered_tokens: {count} (limit {limit}, {'FITS' if count <= limit else 'OVER by ' + str(count - limit)})")
    print(shift_line("reasoning", saved, candidate, "reasoning"))
    print(shift_line("answer", saved, candidate, "answer"))
    print(shift_line("both", saved, candidate, "both"))
    if edited_r == saved_r:
        print("shift[answer, saved reasoning]: same as candidate (reasoning unchanged)")
    else:
        _, fixed = measure(cfg, prompt, saved_r, edited_a)
        print(shift_line("answer, saved reasoning", saved, fixed, "answer"))
    return 0 if count <= limit else 2


if __name__ == "__main__":
    sys.exit(main())
