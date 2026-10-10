"""Save one validated training row and verify the result.

Usage: python authoring/save.py PROMPT.txt REASONING.txt|- ANSWER.txt [--config TOML]

Arguments match count_score.py; pass "-" for an answer-only row. Texts are read
with config.read_prompt and config.read_field, the same readers count_score.py
uses, so the saved row is exactly the text that was scored. The destination is
`destination` in dataset-authoring.toml.

Before saving, an answer that still begins with whitespace is rejected. The
destination's bytes are recorded, the row is appended with `dataset save`, and
the result is verified: the earlier bytes are unchanged, exactly one row was
added, and that row equals the example field for field.

Exit status: 0 saved and verified; 1 invalid input or save failure; 3 the row
was skipped as an exact duplicate (it does not count toward a batch); 4
verification failed. Saves to one destination must run sequentially; a
concurrent append is detected as a verification failure, not prevented.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("reasoning", help="reasoning file, or - for an answer-only row")
    ap.add_argument("answer")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = config.load(args.config)
    dest = pathlib.Path(cfg.get("destination"))
    if not dest.is_absolute():
        dest = config.REPO / dest
    prompt = config.read_prompt(args.prompt)
    reasoning = None if args.reasoning == "-" else config.read_field(args.reasoning)
    answer = config.read_field(args.answer)
    if answer != answer.lstrip():
        raise SystemExit("answer begins with whitespace; the editing contract requires removing it")

    example = {"messages": [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer}]}
    if reasoning is not None:
        example["reasoning"] = reasoning

    before = dest.read_bytes() if dest.exists() else b""
    proc = subprocess.run([sys.executable, "-m", "dataset.cli", "save"], cwd=config.REPO,
                          input=json.dumps({"path": str(dest), "examples": [example]}),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"dataset save failed (exit {proc.returncode}):\n{proc.stderr}")
    result = json.loads(proc.stdout)
    if result.get("duplicates") == 1 and result.get("added") == 0:
        print(f"duplicate: {dest} already contains this exact row; nothing added")
        sys.exit(3)

    after = dest.read_bytes()
    new_lines = after[len(before):].decode("utf-8").splitlines()
    checks = {
        "earlier_bytes_unchanged": after[:len(before)] == before,
        "one_row_added": result.get("added") == 1 and len(new_lines) == 1,
        "row_matches_example": len(new_lines) == 1 and json.loads(new_lines[0]) == example,
    }
    print(f"added={result.get('added')} duplicates={result.get('duplicates')} "
          + " ".join(f"{k}={v}" for k, v in checks.items())
          + f" total_rows={after.count(b'\n')} size={len(after)} sha256={hashlib.sha256(after).hexdigest()}")
    if not all(checks.values()):
        print("VERIFICATION FAILED", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
