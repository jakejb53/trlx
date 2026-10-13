"""Replace or delete exactly one existing row of the destination dataset.

Usage: python authoring/replace_row.py LINE_SHA256 PROMPT.txt REASONING.txt|- ANSWER.txt --destination PATH [--config TOML]
       python authoring/replace_row.py LINE_SHA256 --delete --destination PATH

LINE_SHA256 identifies the row by the SHA256 of its current line, taken as the
line's UTF-8 bytes without the trailing LF. The run fails unless exactly one
line has that hash, so a row that changed since it was read is never touched.
Replacement inputs match save.py: texts are read with config.read_prompt and
config.read_field, "-" means an answer-only row (accepted only when
`allow_answer_only_rows` is true in dataset-authoring.toml; an empty reasoning
file is always an error), and an answer that begins with whitespace is
rejected. The new line uses the serialization of `dataset save` (json.dumps
with ensure_ascii=False; keys messages, reasoning).

The complete new file is computed in memory and checked before publication:
every other line is byte-identical and in order, the row count is unchanged
(one fewer for --delete), and the new line parses to exactly the input fields.
It is published through dataset.io.publish_output (staged, atomic rename), and
the published bytes are read back and compared with the checked bytes.

Exit status: 0 replaced or deleted and verified; 1 invalid input or
publication failure; 4 verification failed. Writes to one destination must be
sequential across processes; a concurrent change is detected, not prevented.
"""
import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config  # noqa: E402

sys.path.insert(0, str(config.REPO))
from dataset.io import publish_output  # noqa: E402


def line_hash(line):
    return hashlib.sha256(line).hexdigest()


def split_rows(data):
    """Lines without their LF, and whether the file ended with one."""
    if not data:
        return [], True
    trailing = data.endswith(b"\n")
    lines = data.split(b"\n")
    if trailing:
        lines.pop()
    return lines, trailing


def build_example(prompt, reasoning, answer):
    example = {"messages": [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer}]}
    if reasoning is not None:
        example["reasoning"] = reasoning
    return example


def rewrite(data, target_hash, new_line=None):
    """Return (new_bytes, index) with the target line replaced, or removed when new_line is None.

    Raises ValueError unless exactly one line has target_hash, then checks the
    result line by line so nothing but the target can differ.
    """
    lines, trailing = split_rows(data)
    hits = [i for i, line in enumerate(lines) if line_hash(line) == target_hash]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} lines have SHA256 {target_hash}; expected exactly 1")
    i = hits[0]
    out = lines[:i] + ([new_line] if new_line is not None else []) + lines[i + 1:]
    new = b"\n".join(out) + (b"\n" if trailing and out else b"")
    check, _ = split_rows(new)
    expected = len(lines) - (0 if new_line is not None else 1)
    if len(check) != expected:
        raise ValueError(f"row count {len(check)} after rewrite, expected {expected}")
    kept = check[:i] + check[i + (1 if new_line is not None else 0):]
    if kept != lines[:i] + lines[i + 1:]:
        raise ValueError("a line other than the target changed")
    if new_line is not None and check[i] != new_line:
        raise ValueError("the replaced line differs from the built row")
    return new, i


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("line_sha256")
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("reasoning", nargs="?", help="reasoning file, or - for an answer-only row")
    ap.add_argument("answer", nargs="?")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--destination", required=True,
                    help="the session's destination JSONL; relative paths use the working directory")
    ap.add_argument("--config", help="path to dataset-authoring.toml; read for the answer-only policy, not the destination")
    args = ap.parse_args()
    texts = [args.prompt, args.reasoning, args.answer]
    if args.delete == any(t is not None for t in texts) or (not args.delete and None in texts):
        raise SystemExit("give either PROMPT REASONING ANSWER or --delete, not both and not a partial set")

    # The destination comes from the session, like `dataset save`'s path, not from
    # dataset-authoring.toml: a session may override the saved destination.
    dest = pathlib.Path(args.destination)
    data = dest.read_bytes()

    new_line = example = None
    if not args.delete:
        # The config is loaded only for a replacement: deletion has no row to check.
        cfg = config.load(args.config)
        prompt = config.read_prompt(args.prompt)
        reasoning = config.read_reasoning(args.reasoning, cfg)
        answer = config.read_field(args.answer)
        if answer != answer.lstrip():
            raise SystemExit("answer begins with whitespace; the editing contract requires removing it")
        example = build_example(prompt, reasoning, answer)
        new_line = json.dumps(example, ensure_ascii=False).encode("utf-8")
        if json.loads(new_line) != example:
            raise SystemExit("built row does not round-trip to the input fields")

    try:
        new, index = rewrite(data, args.line_sha256, new_line)
    except ValueError as e:
        raise SystemExit(f"replace_row: {e}")

    def write(prepared):
        prepared.write_bytes(new)

    publish_output(dest, write, force=True)
    if dest.read_bytes() != new:
        print("verification failed: published bytes differ from the checked rewrite", file=sys.stderr)
        sys.exit(4)
    rows, _ = split_rows(new)
    action = "deleted" if args.delete else "replaced"
    detail = "" if args.delete else f" new_line_sha256={line_hash(rows[index])}"
    print(f"{action} row index={index} total_rows={len(rows)}{detail} sha256={hashlib.sha256(new).hexdigest()}")


if __name__ == "__main__":
    main()
