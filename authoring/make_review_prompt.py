"""Assemble the adversarial reviewer prompt for one row.

Usage: python authoring/make_review_prompt.py RAW.json EDITED_DIR JUSTIFICATIONS.txt OUT.prompt
                                             --prompt-file PROMPT.txt [--prior RULING.txt] [--cuts]

The prompt carries the editing-contract summary, the saved user prompt (read
with config.read_prompt, so it is the exact text generated from, scored, and
saved), the complete unified diffs of
reasoning and answer between untouched and edited texts, the complete untouched
and edited reasoning, the complete edited answer, the editor's justifications,
and the instruction to end with exactly ACCEPT or REVISE. --prior adds the
previous ruling and the editor's response for a resubmission. --cuts is
required whenever material was removed under "Fitting the token limit": it adds
those rules so the reviewer can verify duplication and flow claims against the
whole retained text. Every text is inline; the reviewer reads no files.
"""
import argparse
import difflib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config  # noqa: E402

CONTRACT ="""EDITING CONTRACT (summary)

The editor's task was to repair the target model's actual output while preserving its voice and sound reasoning, not to write a preferred answer. Rules:
1. Before changing any span, identify its specific defect and the fact or explicit requirement it violates. If no defect can be identified, the span is preserved verbatim.
2. Make the smallest correction that resolves the defect. Preserve surrounding sound wording, vocabulary, structure, explanations, examples, and design choices.
3. Permitted edits only: correct a demonstrably false claim, invalid calculation/code, or contradiction; satisfy an explicit prompt requirement or supply something necessary to make a retained claim correct; repair a grammatical error, malformed formatting, or ambiguity that prevents a definite interpretation; remove instructions about composing the response, stale draft commentary, unresolved writing choices, repetition, abandoned deliberation, or duplicate answer drafts while preserving genuine uncertainty about the problem; remove dangling references to excluded system text, Context, tool calls, research steps, internal contracts, or prior authoring turns (the saved row contains only the user prompt, reasoning, and answer; references whose referent exists only in excluded input are drafting residue); fit the verified token limit.
4. Not defects: a different valid design, greater robustness, broader coverage, more formal terminology, or a formulation the editor prefers. Concise outlines or requirement checklists are the model's natural reasoning and are preserved when grammatical, decisive, and consistent with the answer. Prompt planning alone is not a defect.
5. Reasoning and answer must remain consistent. Sound original passages must remain intact. Replacing a whole passage is permitted only when local corrections cannot make it correct and coherent.
6. Remove leading whitespace from the answer.

YOUR TASK

You are the adversarial reviewer. Compare the untouched and edited texts below. For each change, decide whether it is authorized by the contract and minimal. Also check whether any defect the contract requires fixing was left unfixed in the retained text. Check claims about the prompt's facts and requirements against the saved user prompt. Object to any change that rewrites sound text, changes a design choice, summarizes, reorganizes, or exceeds the smallest correction. Then rule.

Write your findings, and on the final line of your output write exactly ACCEPT or exactly REVISE (nothing else on that line). If REVISE, name the exact spans and what local correction is required.
"""

CUTS_RULES = """RULES FOR SIZE CUTS (section "Fitting the token limit")

Material was removed to fit the token limit. Judge each cut by its effect on the demonstration of understanding and on the chain of thought that leads to the response. In the reasoning, only fluff may be removed: filler words, hedging, drafting residue, and repetition of a point already made. Any other reasoning cut is presumed harmful and is permitted only with a span-specific statement, which you must adjudicate, that removing it leaves both the demonstration of understanding and the chain of thought intact. Response cuts remove whole sections from the end; earlier response material may be removed only when it is fluff or duplicates retained content. Retained text may not be rewritten beyond a minimal fix to a reference to a removed section. The retained response must remain correct; reduced completeness is accepted. The complete retained reasoning and response are included below so you can verify duplication and flow claims against the whole retained text.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("edited_dir")
    ap.add_argument("justifications")
    ap.add_argument("out")
    ap.add_argument("--prompt-file", required=True,
                    help="the saved user prompt; required so prompt-dependent corrections can be verified")
    ap.add_argument("--prior", help="file with the prior ruling and the editor's response")
    ap.add_argument("--cuts", action="store_true", help="material was removed under Fitting the token limit")
    args = ap.parse_args()

    raw = json.load(open(args.raw, encoding="utf-8"))
    ed = pathlib.Path(args.edited_dir)
    edited = {f: (ed / f"{f}.txt").read_text(encoding="utf-8") for f in ("reasoning", "answer")}
    parts = [CONTRACT]
    if args.cuts:
        parts.append(CUTS_RULES)
    if args.prior:
        parts.append("PRIOR RULING AND RESPONSE\n\n" + pathlib.Path(args.prior).read_text(encoding="utf-8"))
    parts.append("SAVED USER PROMPT (complete)\n\n" + config.read_prompt(args.prompt_file))
    for f in ("reasoning", "answer"):
        diff = "".join(difflib.unified_diff(raw[f].splitlines(keepends=True),
                                            edited[f].splitlines(keepends=True),
                                            fromfile=f"untouched/{f}", tofile=f"edited/{f}", n=2))
        parts.append(f"COMPLETE UNIFIED DIFF: {f.upper()}\n\n" + (diff or "(no changes)\n"))
    parts.append("UNTOUCHED REASONING (complete)\n\n" + raw["reasoning"])
    parts.append("EDITED REASONING (complete)\n\n" + edited["reasoning"])
    # Always included: the reviewer must check retained text for unfixed defects,
    # and the answer diff alone shows only changed hunks with two lines of context.
    parts.append("EDITED ANSWER (complete)\n\n" + edited["answer"])
    parts.append("EDITOR'S JUSTIFICATIONS\n\n" + pathlib.Path(args.justifications).read_text(encoding="utf-8"))
    out = pathlib.Path(args.out)
    out.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"wrote {out}: {out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
