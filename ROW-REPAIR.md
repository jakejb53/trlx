# Row repair

Follow this procedure when the user specifies a row repair session at the first
onboarding question. The CLAUDE.md approval rules apply, except as authorized
here.

## Setup

Resolve settings as in "Session inputs" in `DATASET-AUTHORING.md`. The
authoring-only questions, the batch workflow, generation, and the adversarial
reviewer do not apply. Then ask which line to start with. Rows are named by
their 1-based line number in the destination; the user tracks progress.

A repair session authorizes creating and editing files under
`authoring-work/repair/`. The destination changes only with the user's approval.

Run commands from the repository root with the project's Python. Settings work
as in an authoring session: use the session's values in each command, including
the session's destination wherever `DESTINATION` appears. `--help` gives each
script's inputs. `$D` below stands for the row's directory: write its literal
path in each command.

## Editing rules

The saved row is the untouched text and the baseline. Edits follow "Edit the
original" in `DATASET-AUTHORING.md`, including "Remove waste"; where it calls for
the reviewer or regeneration, the user's ruling takes their place.

## Each row

1. **Read.** Run the script in "Step 1 script", replacing `DESTINATION` and `N`. It writes
   `prompt.txt`, `raw.json`, `raw.reasoning.txt`, `raw.answer.txt`, and
   `line.sha256` into `authoring-work/repair/line-N-<hash prefix>/` and prints
   that directory. Keep these files: after a replacement they are the only copy
   of the row.
2. **Analyze.** Read `REASONING-EDIT-REMINDER.md`, then the whole row.
   Label the waste in `$D/labels.json`. Each repetition label's reason names the
   earlier passage it repeats. Each beyond-prompt label's reason names the
   prompt's requests and states that the section serves none of them. When a
   label's text is not unique, give it an `anchor` that extends into the
   following words. When a removal leaves a
   sentence starting in lowercase, give that label an anchor that extends into
   the next word, even if its text is unique, and add the anchor to
   `$D/capitalize.json` (a list; `[]` if none). Then run
   `python authoring/span_report.py $D/prompt.txt $D/raw.json $D/labels.json --edits-out $D/removals.json`.
   Review the row for correctness: for each suspected defect, record the span,
   the fact or requirement it violates, and whether it was checked against a
   source or remains unverified. Do not fix anything yet. Draft how to validate
   the row's correctness with what is available.
3. **Present and decide.** Show the prompt, reasoning, and answer verbatim
   (fenced, per `DATASET-AUTHORING.md`), then the `span_report.py` output, each
   label with its reason, the correctness findings, and the proposed
   validation. Wait for the user's decision: keep, delete, or repair. Keep
   leaves the row unchanged; delete runs step 8's delete command; repair
   continues with step 4 using the validation plan as agreed or amended by the
   user.
4. **Validate.** Prefer execution; run code only in `$D` or a container. Record
   what was executed, what was checked against sources, and what remains
   unverified.
5. **Repair.** Put corrections in `$D/fixes.json` (`[]` if none), in `edit.py`'s
   format. Build the edit list and apply it with the commands in "Build
   commands". After changing the labels, run step 2's `span_report.py` command
   first; after any change to the labels, `capitalize.json`, or fixes, run the
   build commands again. Read the edited text and re-run the correctness checks.
6. **Measure.** Run `python authoring/count_score.py PROMPT REASONING ANSWER --score`
   for the saved row (`$D/prompt.txt $D/raw.reasoning.txt $D/raw.answer.txt`),
   the candidate (`$D/prompt.txt $D/edited/reasoning.txt $D/edited/answer.txt`),
   and the saved reasoning with the edited answer. Report the token counts
   against the limit and the log-probability differences; no threshold applies.
7. **Present and wait.** Report the waste removals, the correctness findings
   and fixes, what was validated, the measurements, and the path of
   `$D/diff.txt`, and show the candidate's prompt, reasoning, and answer
   verbatim (fenced, per `DATASET-AUTHORING.md`). Replacing a row drops
   trailing newlines its fields may end
   with; the template ignores them. Wait for the user's ruling: replace, delete,
   or leave the row unchanged.
8. **Carry out the ruling.** If it changes the candidate, rebuild, re-measure,
   present, and wait for approval. A prompt change also changes the candidate:
   write the new prompt to `$D/edited/prompt.txt` and use it in place of
   `$D/prompt.txt` when measuring and replacing. A ruling to replace or delete
   is the approval; then run
   `python authoring/replace_row.py $(cat $D/line.sha256) $D/prompt.txt $D/edited/reasoning.txt $D/edited/answer.txt --destination DESTINATION`
   or `python authoring/replace_row.py $(cat $D/line.sha256) --delete --destination DESTINATION`, and
   report its output. After a deletion, later rows move up one line. After any
   ruling, including keep or delete at step 3, ask only whether to move on to
   the next row.

## Step 1 script

Copy it unindented and replace `DESTINATION` with the session's destination and
`N` with the line number.

```sh
python - DESTINATION N <<'EOF'
import hashlib, json, pathlib, sys
dest = pathlib.Path(sys.argv[1])
n = int(sys.argv[2])
line = dest.read_bytes().split(b"\n")[n - 1]
h = hashlib.sha256(line).hexdigest()
row = json.loads(line)
prompt, answer, reasoning = row["messages"][0]["content"], row["messages"][1]["content"], row["reasoning"]
d = pathlib.Path(f"authoring-work/repair/line-{n}-{h[:8]}")
d.mkdir(parents=True)
(d / "prompt.txt").write_text(prompt, encoding="utf-8")
(d / "raw.reasoning.txt").write_text(reasoning, encoding="utf-8")
(d / "raw.answer.txt").write_text(answer, encoding="utf-8")
(d / "line.sha256").write_text(h, encoding="utf-8")
(d / "raw.json").write_text(json.dumps({"reasoning": reasoning, "answer": answer}, ensure_ascii=False), encoding="utf-8")
print(d)
EOF
```

## Build commands

Copy them unindented and replace `$D` with the row's directory.

```sh
python - $D <<'EOF'
import json, sys
d = sys.argv[1]
removals = json.load(open(f"{d}/removals.json"))
for anchor in json.load(open(f"{d}/capitalize.json")):
    [e] = [e for e in removals if e["old"] == anchor]
    e["new"] = e["new"][:1].upper() + e["new"][1:]
json.dump(removals + json.load(open(f"{d}/fixes.json")), open(f"{d}/edits.json", "w"), indent=1, ensure_ascii=False)
EOF
python authoring/edit.py $D/raw.json $D/edits.json $D/edited > $D/diff.txt
```
